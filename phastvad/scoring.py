"""Window scores of PHaST-VAD (numpy only).

Per window t (all read-outs clipped to [1e-8, 1]):
    p_t = P(No | "Is this clip normal?")            normalised over {Yes, No}
    a_t = P(A | state question)                     normalised over {A, B, C, D}
    e_t = max(a_t, a_{t-1}, ..., a_{t-K})           K = 3, memory restarts with every video / stream
    h_t = sqrt(a_t * e_t)
    PH_t = sqrt(p_t * h_t)                          main score
Optional q probe (--with-q):
    q_t = P(Yes | "Is an abnormal event happening in this clip?")
    QH_t = sqrt(q_t * h_t),  QPH_t = (q_t * p_t * h_t)^(1/3)
Diagnostics: pa_t = sqrt(p_t * a_t).
Only the current and past windows are used (window-level causal: a score is emitted once its window has been observed).
Each window score is assigned to the LAST 2 s of its window, so a frame score can depend on content up to ~1.5 s
after that frame (~3.5 s in the first 2 s of a video); offline frame metrics are not frame-level causal predictions.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import EPS, MEMORY_K, WINDOW_SEC

SCORES = ('PH', 'QH', 'QPH', 'p', 'a', 'h', 'pa', 'q')
NEEDS_Q = ('QH', 'QPH', 'q')


def clip(x) -> np.ndarray:
    return np.clip(np.asarray(x, np.float64), EPS, 1.0)


def memory(a: np.ndarray, k: int = MEMORY_K) -> np.ndarray:
    """e_t = max(a_t .. a_{t-k}) within one video (fewer terms at the video start)."""
    a = np.asarray(a, np.float64)
    return np.array([a[max(0, t - k):t + 1].max() for t in range(len(a))]) if len(a) else a.copy()


def scores(p, state, q=None, k: int = MEMORY_K) -> dict[str, np.ndarray]:
    """All score variants of one video from its window read-outs p [W], state [W, 4] and optional q [W]."""
    p, a = clip(p), clip(np.asarray(state, np.float64).reshape(-1, 4)[:, 0])
    e = memory(a, k)
    h = np.sqrt(a * e)
    S = {'p': p, 'a': a, 'e': e, 'h': h, 'pa': np.sqrt(p * a), 'PH': np.sqrt(p * h)}
    if q is not None:
        q = clip(q)
        S.update(q=q, QH=np.sqrt(q * h), QPH=np.cbrt(q * p * h))
    return S


class OnlinePH:
    """Streaming version of `scores`: call once per window, in order; keeps only the last K values of a."""

    def __init__(self, k: int = MEMORY_K):
        self.past = deque(maxlen=k) if k > 0 else None

    def __call__(self, p: float, state, q: float | None = None) -> dict[str, float]:
        p, a = float(clip(p)), float(clip(state[0]))
        e = max([a, *(self.past or [])])
        if self.past is not None:
            self.past.append(a)
        h = float(np.sqrt(a * e))
        out = {'p': p, 'a': a, 'e': e, 'h': h, 'pa': float(np.sqrt(p * a)), 'PH': float(np.sqrt(p * h))}
        if q is not None:
            q = float(clip(q))
            out.update(q=q, QH=float(np.sqrt(q * h)), QPH=float(np.cbrt(q * p * h)))
        return out


def window_to_frames(num_frames: int, fps: float, starts, window_scores, window_sec: float = WINDOW_SEC) -> np.ndarray:
    """Frame scores: window [s, s + 4) scores frames [floor((s + 2) fps), ceil((s + 4) fps)); overlaps are averaged,
    uncovered frames (the first ~2 s) are linearly interpolated (constant extrapolation)."""
    starts = np.asarray(starts, np.float64)
    w = np.asarray(window_scores, np.float64)
    n = int(num_frames)
    if n <= 0 or len(w) == 0:
        return np.zeros(max(n, 0), np.float64)
    left = np.minimum(n - 1, np.maximum(0, np.floor((starts + window_sec / 2) * fps).astype(np.int64)))
    right = np.minimum(n, np.ceil((starts + window_sec) * fps).astype(np.int64))
    right = np.where(right <= left, np.minimum(n, left + 1), right)
    sums, counts = np.zeros(n + 1), np.zeros(n + 1, np.int64)
    np.add.at(sums, left, w)
    np.add.at(sums, right, -w)
    np.add.at(counts, left, 1)
    np.add.at(counts, right, -1)
    sums, counts = np.cumsum(sums)[:n], np.cumsum(counts)[:n]
    covered = counts > 0
    out = np.empty(n, np.float64)
    out[covered] = sums[covered] / counts[covered]
    idx = np.arange(n)
    out[~covered] = np.interp(idx[~covered], idx[covered], out[covered])
    return out
