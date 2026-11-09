"""Video input: decord clip reader for files, window grid, window types, and an OpenCV live capture.

Window w covers [s_w, s_w + 4 s); its 8 frames are at s_w + i * 0.5 s (i = 0..7), source frame index
rint(t * source_fps) clipped to the video. Grid of a file of duration D: s = 0, 2, 4, ... up to D - 4, plus a final
window aligned to the end (s = D - 4) when the regular grid does not reach it ("terminal" window).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (import torch before decord: avoids native crashes on some servers)

from .config import CLIP_FRAMES, STRIDE_SEC, WINDOW_SEC


def clip_indices(start_sec: float, fps: float, num_frames: int | None = None) -> list[int]:
    times = start_sec + np.arange(CLIP_FRAMES) * (WINDOW_SEC / CLIP_FRAMES)
    idx = np.rint(times * fps).astype(np.int64)
    return (np.clip(idx, 0, num_frames - 1) if num_frames else np.maximum(idx, 0)).tolist()


def make_window_starts(duration_sec: float) -> np.ndarray:
    max_start = max(0.0, duration_sec - WINDOW_SEC)
    starts = np.arange(0.0, max_start + 1e-9, STRIDE_SEC, dtype=np.float64)
    if len(starts) == 0:
        starts = np.array([0.0])
    if starts[-1] < max_start - 1e-6:
        starts = np.append(starts, max_start)
    return starts


def window_types(starts) -> np.ndarray:
    """'cold' = first window of a video / stream (decodes all 8 frames; opening the file is not included, see open_s), 'terminal' = final window
    aligned to the video end (arrives < 2 s after the previous one), 'regular' = every other window (2 s cadence).
    Latency statistics are reported per window type."""
    starts = np.asarray(starts, np.float64)
    t = np.full(len(starts), 'regular', dtype=object)
    if len(t):
        t[0] = 'cold'
    if len(t) > 1 and starts[-1] - starts[-2] < STRIDE_SEC - 1e-3:
        t[-1] = 'terminal'
    return t


class ClipReader:
    """Random-access decoder of a video file (decord, CPU) with a small buffer of decoded frames: consecutive windows
    share 4 frames, so only the new ones are decoded."""

    def __init__(self, path: Path | str):
        from decord import VideoReader, cpu

        self.path = Path(path)
        self.reader = VideoReader(str(path), num_threads=-1, ctx=cpu(0))
        self.num_frames = len(self.reader)
        self.fps = float(self.reader.get_avg_fps())
        if self.num_frames <= 0 or not np.isfinite(self.fps) or self.fps <= 0:
            raise RuntimeError(f'invalid video: {path}')
        self._buf: dict[int, np.ndarray] = {}

    @property
    def duration_sec(self) -> float:
        return self.num_frames / self.fps

    def window_starts(self) -> np.ndarray:
        return make_window_starts(self.duration_sec)

    def clip(self, start_sec: float) -> tuple[np.ndarray, list[int], int]:
        """(frames [8, H, W, 3] RGB uint8, source frame indices, number of newly decoded frames)."""
        keys = clip_indices(start_sec, self.fps, self.num_frames)
        new = sorted({k for k in keys if k not in self._buf})
        if new:
            self._buf.update(zip(new, self.reader.get_batch(new).asnumpy()))
        frames = np.stack([self._buf[k] for k in keys])
        self._buf = {k: self._buf[k] for k in set(keys)}
        return frames, keys, len(new)


class LiveCapture:
    """Live source (camera index, RTSP/HTTP URL or a file played at its own frame rate) read by a background thread.

    Frames are numbered in arrival order; frame n is taken as time n / fps (fps = the source's nominal rate or
    `fps`), so the window rule of files applies unchanged. Window w is complete when the frame at s_w + 4 s has
    arrived; its arrival time is the wall-clock time of that frame. Only the last `keep_sec` seconds are kept in
    memory: a window whose frames were already dropped (processing fell that far behind) is returned without frames.
    """

    def __init__(self, source: str, fps: float | None = None, keep_sec: float = 20.0):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.cap.isOpened():
            raise RuntimeError(f'cannot open live source {source}')
        nominal = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps or (nominal if nominal and np.isfinite(nominal) and nominal > 0 else 25.0))
        self.is_file = Path(source).is_file()
        self.keep = int(np.ceil(keep_sec * self.fps)) + 1
        self.frames: dict[int, tuple[float, np.ndarray]] = {}
        self.count, self.ended, self.t0, self._stop = 0, False, None, False
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read()
            now = time.perf_counter()
            if not ok:
                break
            if self.t0 is None:
                self.t0 = now
            if self.is_file:                      # emulate a camera: deliver frame n at t0 + n / fps
                wait = self.t0 + self.count / self.fps - now
                if wait > 0:
                    time.sleep(wait)
                now = time.perf_counter()
            rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            with self.cond:
                self.frames[self.count] = (now, rgb)
                self.frames.pop(self.count - self.keep, None)
                self.count += 1
                self.cond.notify_all()
        with self.cond:
            self.ended = True
            self.cond.notify_all()
        self.cap.release()

    def wait_window(self, start_sec: float) -> tuple[float, np.ndarray | None, list[int]] | None:
        """Block until window [start, start + 4 s) is complete -> (arrival time, frames or None if already dropped,
        frame indices); None at the end of the stream."""
        last = int(np.ceil((start_sec + WINDOW_SEC) * self.fps)) - 1
        keys = clip_indices(start_sec, self.fps)
        with self.cond:
            self.cond.wait_for(lambda: self.count > last or self.ended)
            if self.count <= last:
                return None
            arrival = self.frames[last][0] if last in self.frames else time.perf_counter()
            if any(k not in self.frames for k in keys):
                return arrival, None, keys
            return arrival, np.stack([self.frames[k][1] for k in keys]), keys

    def close(self) -> None:
        self._stop = True
        self.thread.join(timeout=5)
