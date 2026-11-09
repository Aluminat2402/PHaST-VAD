#!/usr/bin/env python3
"""Write the UBnormal test annotation in the format read by score_videos.py (no model needed).

UBnormal (Acsintoae et al., CVPR 2022) ships per-frame masks and, for abnormal videos, a tracks file
<video>_annotations/<video>_tracks.txt with lines "object_id,start_frame,end_frame". Following the official
write_frame_level_ground_truth.py (github.com/lilygeorgescu/UBnormal), a frame is abnormal if it lies in
[start, end] of any track; normal videos are all-normal. The official test split (158 abnormal + 53 normal videos)
is in splits/ubnormal/.

Output: <file.mp4> <Abnormal|Normal> s1 e1 [s2 e2 ...]  (0-based inclusive frame indices, -1 padding)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path, help='UBnormal root containing Scene1 ... Scene29')
    parser.add_argument('--output', type=Path, default=None, help='Default: <root>/annotations/test_annotation.txt')
    args = parser.parse_args()
    output = args.output or args.root / 'annotations/test_annotation.txt'
    lines, missing = [], []
    for kind in ('abnormal', 'normal'):
        for name in sorted(np.loadtxt(HERE / f'splits/ubnormal/{kind}_test_video_names.txt', dtype=str)):
            scene = args.root / f'Scene{name.split("_")[2]}'
            if not (scene / f'{name}.mp4').is_file():
                missing.append(name)
                continue
            fields = ['-1', '-1', '-1', '-1']
            if kind == 'abnormal':
                tracks = np.loadtxt(scene / f'{name}_annotations' / f'{name}_tracks.txt', delimiter=',', ndmin=2)
                flat = [str(x) for iv in merge([(int(t[1]), int(t[2])) for t in tracks]) for x in iv]
                fields = flat + ['-1', '-1'] * (len(flat) == 2)
            lines.append(f'{name}.mp4 {"Abnormal" if kind == "abnormal" else "Normal"} ' + ' '.join(fields))
    if missing:
        raise SystemExit(f'{len(missing)} test videos not found under {args.root}: {missing[:5]}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n')
    print(f'{len(lines)} test videos -> {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
