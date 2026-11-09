"""Window loop shared by offline scoring and streaming replay, per-window records and latency statistics.

Wall-clock replay of a file (the protocol of the reported runtime numbers, one stream on one GPU):
  window w of a video arrives at arrival_w = s_w + 4 s on the wall clock (t = 0 when the stream starts, after the
  file is opened); its deadline is arrival_w + 2 s (the next window). The loop sleeps until the arrival, decodes the
  new frames, runs the probes and the online score. Per window it records
      arrival, start (processing start), finish (score available)       seconds since stream start
      latency = finish - arrival (end to end),  queue = start - arrival,  processing = finish - start
      stage times: decode, preprocess, vision, prefill (shared prefix), suffix (probe batch + read-out), score.
Fast replay (pace='none'): no waiting, arrival := start, so latency = processing time (throughput mode, not a
real-time measurement). Offline scoring is the same loop with pace='none'.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np

from .config import DEADLINE_SEC, WINDOW_SEC
from .scoring import OnlinePH
from .video import window_types

STAGES = ('decode', 'preprocess', 'vision', 'prefill', 'suffix', 'score', 'total')


def process_windows(model, windows: Iterable, pace: str = 'realtime', on_window: Callable | None = None,
                    check: int = 0) -> list[dict]:
    """Run the probes and the online score over `windows`, an iterable of (start_sec, getter) where getter() returns
    (frames, frame_indices, new_frames, arrival or None). pace='realtime' waits until s + 4 s on the wall clock
    (file replay); with pace='none' or an explicit arrival time (live capture) it does not wait."""
    model.reset()
    online = OnlinePH()
    rows = []
    wall0 = time.perf_counter()
    for w, (start_sec, get) in enumerate(windows):
        arrival = float(start_sec) + WINDOW_SEC
        if pace == 'realtime':
            wait = arrival - (time.perf_counter() - wall0)
            if wait > 0:
                time.sleep(wait)
        ts = time.perf_counter()
        frames, keys, new, live_arrival = get()
        if frames is None:                                   # live capture fell behind: frames already dropped
            rows.append({'window': w, 'start_sec': float(start_sec), 'dropped': True})
            continue
        td = time.perf_counter()
        tm = {}
        r = model(frames, keys, tm)
        t5 = time.perf_counter()
        s = online(r['p'], r['state'], r.get('q'))
        tf = time.perf_counter()
        if live_arrival is not None:
            arrival = live_arrival - wall0
        elif pace != 'realtime':
            arrival = ts - wall0
        row = {'window': w, 'start_sec': float(start_sec), 'frame_indices': keys, 'arrival': arrival,
               'start': ts - wall0, 'finish': tf - wall0, 't_decode': td - ts, 't_preprocess': tm['preprocess'],
               't_vision': tm['vision'], 't_prefill': tm['prefill'], 't_suffix': tm['suffix'], 't_score': tf - t5,
               't_total': tf - ts, 'new_frames': new, 'new_vision_frames': tm['new_vision_frames'],
               'tokens_prefix': tm['tokens_prefix'], **r, **{f'score_{k}': v for k, v in s.items()}}
        if w < check:
            ref = model.reference(frames)
            row['check'] = [abs(ref['p'] - r['p']), float(np.abs(ref['state'] - r['state']).max()),
                            abs(ref['q'] - r['q']) if 'q' in r else np.nan]
        rows.append(row)
        if on_window is not None:
            on_window(row)
    return rows


def file_windows(reader, starts) -> Iterable:
    for s in starts:
        yield s, (lambda s=s: (*reader.clip(s), None))


def live_windows(capture, max_windows: int | None = None) -> Iterable:
    from .config import STRIDE_SEC
    w = 0
    while max_windows is None or w < max_windows:
        s = w * STRIDE_SEC
        got = capture.wait_window(s)
        if got is None:
            return
        arrival, frames, keys = got
        yield s, (lambda f=frames, k=keys, a=arrival: (f, k, len(k), a))
        w += 1


def to_arrays(rows: list[dict], with_q: bool) -> dict[str, np.ndarray]:
    """Per-window records -> arrays for the .npz output (dropped live windows are left out)."""
    rows = [r for r in rows if not r.get('dropped')]
    A = lambda k, dt=np.float64: np.array([r[k] for r in rows], dt)
    out = {'window_starts_sec': A('start_sec'), 'frame_indices': A('frame_indices', np.int64).reshape(-1, 8),
           'p': A('p'), 'state': A('state').reshape(-1, 4), 'lp_p': A('lp_p').reshape(-1, 2),
           'lp_state': A('lp_state').reshape(-1, 4), 'arrival': A('arrival'), 'start': A('start'), 'finish': A('finish'),
           'new_frames': A('new_frames', np.int16), 'new_vision_frames': A('new_vision_frames', np.int16)}
    out.update({f't_{k}': A(f't_{k}') for k in STAGES})
    for k in ('PH', 'p', 'a', 'e', 'h', 'pa') + (('q', 'QH', 'QPH') if with_q else ()):
        out[f'score_{k}'] = A(f'score_{k}')
    if with_q:
        out['q'], out['lp_q'] = A('q'), A('lp_q').reshape(-1, 2)
    if rows and 'check' in rows[0]:
        out['check'] = np.array([r['check'] for r in rows if 'check' in r], np.float64)
    return out


def _stats(x) -> dict:
    x = np.asarray(x, np.float64)
    if not len(x):
        return {'windows': 0}
    return {'windows': int(len(x)), 'mean': float(x.mean()), 'p50': float(np.percentile(x, 50)),
            'p95': float(np.percentile(x, 95)), 'p99': float(np.percentile(x, 99)), 'max': float(x.max()),
            'miss_pct_gt2s': 100.0 * float(np.mean(x > DEADLINE_SEC)), 'misses': int(np.sum(x > DEADLINE_SEC))}


def latency_summary(videos: list[dict]) -> dict:
    """Latency (finish - arrival) by window type over one or more videos (dicts with window_starts_sec, arrival,
    finish, start, t_* arrays): all / cold / regular / terminal / steady (regular + terminal), plus queue delay,
    processing time and stage means."""
    acc: dict[str, dict[str, list]] = {}
    for v in videos:
        types = window_types(v['window_starts_sec'])
        e2e = v['finish'] - v['arrival']
        proc = v['finish'] - v['start']
        queue = v['start'] - v['arrival']
        for i, t in enumerate(types):
            for k in ('all', t) + (('steady (regular + terminal)',) if t != 'cold' else ()):
                a = acc.setdefault(k, {'e2e': [], 'proc': [], 'queue': [], **{s: [] for s in STAGES}})
                a['e2e'].append(e2e[i]); a['proc'].append(proc[i]); a['queue'].append(queue[i])
                for s in STAGES:
                    a[s].append(v[f't_{s}'][i])
    out = {}
    for k in ('all', 'cold', 'regular', 'terminal', 'steady (regular + terminal)'):
        if k in acc:
            a = acc[k]
            out[k] = {'latency': _stats(a['e2e']), 'processing_mean': float(np.mean(a['proc'])),
                      'rtf_mean': float(np.mean(a['proc']) / DEADLINE_SEC), 'queue_p99': float(np.percentile(a['queue'], 99)),
                      'stage_mean': {s: float(np.mean(a[s])) for s in STAGES}}
    return out


def format_summary(summary: dict) -> str:
    lines = [f'{"window type":30s} {"windows":>8s} {"p50":>7s} {"p95":>7s} {"p99":>7s} {"max":>7s} {"miss>2s":>8s} {"RTF":>6s}']
    for k, s in summary.items():
        L = s['latency']
        lines.append(f'{k:30s} {L["windows"]:8d} {L["p50"]:7.3f} {L["p95"]:7.3f} {L["p99"]:7.3f} {L["max"]:7.3f} '
                     f'{L["miss_pct_gt2s"]:7.2f}% {s["rtf_mean"]:6.3f}')
    return '\n'.join(lines)
