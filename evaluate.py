#!/usr/bin/env python3
"""Evaluate a scores/ directory written by score_videos.py or stream.py with --annotation (CPU, no model).

Window scores are recomputed from the stored read-outs (p, state and, when it was run, q), mapped to frames
(last 2 s of every window, mean over overlaps, interpolation elsewhere) and scored against the stored frame labels:
  micro AUC / AP (main), macro AUC / AP, video AUC, normal-frame FPR@0.5, padded macro AUC (diagnostic);
  definitions in phastvad/metrics.py.
  --score PH (default) | QH | QPH | p | a | h | pa | q        (QH, QPH, q need outputs scored with --with-q)
  --video-list FILE   evaluate a subset (e.g. MSAD protocol ii, written by prepare_msad.py)

  python evaluate.py out/ucf/scores [--score PH] [--output metrics.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phastvad.datasets import read_list
from phastvad.metrics import KEYS, evaluate
from phastvad.scoring import NEEDS_Q, SCORES, scores, window_to_frames

NAMES = {'mauc': 'micro AUC', 'ap': 'micro AP', 'macauc': 'macro AUC', 'macap': 'macro AP', 'vidauc': 'video AUC',
         'nfpr': 'normal-frame FPR@0.5', 'offmac': 'padded macro AUC'}


def load(scores_dir: Path, score: str, video_list: Path | None = None):
    keep = read_list(video_list)
    frames, labels, cats = [], [], []
    for path in sorted(scores_dir.glob('*.npz')):
        if keep is not None and path.stem not in keep:
            continue
        with np.load(path, allow_pickle=False) as z:
            if 'frame_labels' not in z.files:
                raise SystemExit(f'{path.name} has no labels; score with --annotation to evaluate')
            if score in NEEDS_Q and 'q' not in z.files:
                raise SystemExit(f'{path.name} has no q read-out; score with --with-q to evaluate {score}')
            S = scores(z['p'], z['state'], z['q'] if 'q' in z.files else None)
            frames.append(window_to_frames(len(z['frame_labels']), float(z['source_fps']), z['window_starts_sec'], S[score]))
            labels.append(z['frame_labels'])
            cats.append(str(z['category']))
    if keep is not None and len(frames) != len(keep):
        print(f'NOTE: {len(frames)} of {len(keep)} listed videos found in {scores_dir}')
    if not frames:
        raise SystemExit(f'no score files in {scores_dir}')
    return frames, labels, cats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scores_dir', type=Path)
    ap.add_argument('--score', choices=SCORES, default='PH')
    ap.add_argument('--video-list', type=Path, default=None)
    ap.add_argument('--output', type=Path, default=None, help='also write the metrics to this JSON file')
    args = ap.parse_args()
    m = evaluate(*load(args.scores_dir, args.score, args.video_list))
    m['score'] = args.score
    print(f'score {args.score}: {m["videos"]} videos ({m["normal_videos"]} normal, {m["macro_videos"]} in macro), '
          f'{m["frames"]} frames ({m["abnormal_frames"]} abnormal)')
    for k in KEYS:
        print(f'  {NAMES[k]:22s} {"-" if m[k] is None else f"{m[k]:.2f}"}')
    if args.output:
        args.output.write_text(json.dumps(m, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
