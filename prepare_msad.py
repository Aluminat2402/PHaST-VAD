#!/usr/bin/env python3
"""Write the MSAD test annotation in the format read by score_videos.py (no model, no video decoding).

Expects the MSAD release layout under <root>:
  MSAD_anomaly/<Type>/<name>.mp4             240 abnormal videos (all are test videos)
  MSAD_normal_testing_videos/<name>.mp4      120 normal test videos
  anomaly_annotation.csv                     name, scenario, total frames, starting / ending frame of anomaly
                                             (1-based, inclusive)
  MSAD_I3D_WS_Test.list                      weakly-supervised test list (protocol ii)
Writes
  <out>/msad_test_annotation.txt   protocol i (360 videos): "<name>.mp4 <Type> <s-1> <e-1> T=<total frames>" or
                                   "<name>.mp4 Normal -1 -1"; T= maps the interval by relative position when the
                                   decoded frame count differs from the annotated one (phastvad/datasets.py)
  <out>/msad_protocol_ii.txt       protocol ii video names (subset of protocol i), for evaluate.py --video-list
Nothing from the dataset is copied; keep MSAD videos, frames and these files out of public repositories.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', type=Path, help='MSAD root (see above)')
    ap.add_argument('--out', type=Path, required=True, help='output directory for the two text files')
    args = ap.parse_args()
    anomaly = {p.stem: p for p in sorted((args.root / 'MSAD_anomaly').glob('*/*.mp4'))}
    normal = {p.stem: p for p in sorted((args.root / 'MSAD_normal_testing_videos').glob('*.mp4'))}
    ann = {r['name']: r for r in csv.DictReader((args.root / 'anomaly_annotation.csv').open())}
    ws = [re.sub(r'_i3d\.npy$', '', l.strip().split('/')[-1])
          for l in (args.root / 'MSAD_I3D_WS_Test.list').open() if l.strip()]
    problems = []
    if set(anomaly) & set(normal):
        problems.append('duplicate names between abnormal and normal videos')
    problems += [f'no annotation for abnormal video {s}' for s in anomaly if s not in ann]
    problems += [f'protocol ii video {s} not found' for s in ws if s not in anomaly and s not in normal]
    if problems:
        raise SystemExit('\n'.join(problems[:10]))
    lines = []
    for s, p in anomaly.items():
        r = ann[s]
        start, end, total = int(r['starting frame of anomaly']), int(r['ending frame of anomaly']), int(r['total frames'])
        lines.append(f'{s}.mp4 {p.parent.name} {start - 1} {end - 1} T={total}')
    lines += [f'{s}.mp4 Normal -1 -1' for s in sorted(normal, key=lambda s: (len(s), s))]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'msad_test_annotation.txt').write_text('\n'.join(lines) + '\n')
    (args.out / 'msad_protocol_ii.txt').write_text('\n'.join(sorted(set(ws))) + '\n')
    print(f'protocol i: {len(anomaly)} abnormal + {len(normal)} normal videos; protocol ii: {len(set(ws))} videos -> {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
