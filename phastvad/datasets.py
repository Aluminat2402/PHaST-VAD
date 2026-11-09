"""Annotation files and frame labels (numpy only).

Annotation format (one video per line, the UCF-Crime test format):
    <file.mp4> <category> <start> <end> [<start> <end> ...] [T=<annotated frames>]
  - start / end: 0-based, inclusive frame indices of the abnormal intervals; -1 = unused; category `Normal` for
    normal videos.
  - optional `T=<n>` (written by prepare_msad.py): the intervals refer to n annotated frames. When the decoded video
    has a different number of frames N, decoded frame j is abnormal iff start / n <= (j + 0.5) / N < (end + 1) / n
    (same time span; identical to plain indices when N == n).
Labels are never passed to the model; they are only stored next to the scores for evaluation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_annotations(path: Path) -> dict[str, dict]:
    """file name -> {'category': str, 'intervals': [(start, end), ...], 'total': int | None}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        total = None
        if fields[-1].startswith('T='):
            total = int(fields.pop()[2:])
        values = [int(v) for v in fields[2:]]
        intervals = [(s, e) for s, e in zip(values[0::2], values[1::2]) if s >= 0 and e >= 0]
        out[fields[0]] = {'category': fields[1], 'intervals': intervals, 'total': total}
    return out


def make_labels(num_frames: int, intervals, total: int | None = None) -> np.ndarray:
    labels = np.zeros(num_frames, np.uint8)
    if total is not None and total != num_frames:
        pos = (np.arange(num_frames) + 0.5) / num_frames
        for s, e in intervals:
            labels |= ((pos >= s / total) & (pos < (e + 1) / total)).astype(np.uint8)
        return labels
    for s, e in intervals:
        labels[max(0, s):min(num_frames, e + 1)] = 1
    return labels


def read_list(path: Path | None) -> set[str] | None:
    """Optional video list (one stem or file name per line, '#' comments) -> set of stems."""
    if path is None:
        return None
    return {Path(l.split()[0]).stem for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith('#')}
