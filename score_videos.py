#!/usr/bin/env python3
"""Offline scoring with PHaST-VAD: one .npz per video in <output-dir>/scores/ (no waiting, one GPU).

Every .mp4 under --video-dir (recursive) is scored, or with --annotation only the annotated videos, whose frame labels
are then stored for evaluate.py (labels never reach the model). Finished videos are skipped on restart;
--num-shards / --shard-index split the videos over several GPUs (one process per GPU).

Scores are window-level causal: window t only sees its own 8 frames and the state memory of earlier windows, so
offline scoring and streaming replay (stream.py) use the same engine and read-outs; only the timing differs.

  python score_videos.py --video-dir V --annotation A.txt --output-dir out/ucf [--with-q]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from phastvad import config as cfg
from phastvad.datasets import make_labels, parse_annotations, read_list
from phastvad.model import PHModel
from phastvad.pipeline import file_windows, process_windows, to_arrays
from phastvad.scoring import window_to_frames
from phastvad.video import ClipReader


def save_npz(path: Path, **values) -> None:
    tmp = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, **values)
    os.replace(tmp, path)


def select_videos(args) -> tuple[list[str], dict[str, Path], dict | None]:
    """Annotated (or all) .mp4 file names after --video-list / --only-video / sharding / --max-videos."""
    found: dict[str, list[Path]] = {}
    for p in sorted(args.video_dir.rglob('*.mp4')):
        found.setdefault(p.name, []).append(p)
    ann = parse_annotations(args.annotation) if args.annotation else None
    names = sorted(ann) if ann is not None else sorted(found)
    keep = read_list(args.video_list)
    if keep is not None:
        names = [n for n in names if Path(n).stem in keep]
    if args.only_video:
        only = set(args.only_video)
        names = [n for n in names if n in only or Path(n).stem in only]
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f'{len(missing)} annotated videos not found under {args.video_dir}: {missing[:5]}')
    dup = [n for n in names if len(found[n]) > 1]
    if dup:
        raise SystemExit(f'{len(dup)} selected file names occur more than once under {args.video_dir}: {dup[:3]}')
    names = names[args.shard_index::args.num_shards][:args.max_videos]
    return names, {n: found[n][0] for n in names}, ann


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument('--video-dir', type=Path, help='searched recursively for .mp4 files')
    ap.add_argument('--annotation', type=Path, default=None, help='annotation file (format: phastvad/datasets.py)')
    ap.add_argument('--video-list', type=Path, default=None, help='only these videos (one name per line)')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--with-q', action='store_true', help='also run the q probe (adds QH, QPH; one more probe)')
    ap.add_argument('--hf-cache', type=Path, default=os.environ.get('HF_HUB_CACHE'), help='HuggingFace cache dir')
    ap.add_argument('--model-revision', default=cfg.MODEL_REVISION, help='pinned model revision (default: frozen)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--only-video', action='append', default=[], help='file name or stem (repeatable)')
    ap.add_argument('--max-videos', type=int, default=None)
    ap.add_argument('--max-windows', type=int, default=None, help='smoke test: first N windows of every video')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--check', type=int, default=0, help='compare the fast engine with reference() on N windows')


def video_record(name: str, reader: ClipReader, rows: list[dict], with_q: bool, ann: dict | None) -> dict:
    """Arrays stored per video: read-outs, scores, timings, PH frame scores and (with an annotation) labels."""
    arr = to_arrays(rows, with_q)
    rec = {'filename': np.array(name), 'source_fps': np.float64(reader.fps), 'num_frames': np.int64(reader.num_frames),
           'with_q': np.bool_(with_q), **arr,
           'frame_scores': window_to_frames(reader.num_frames, reader.fps, arr['window_starts_sec'],
                                            arr['score_PH']).astype(np.float32)}
    if ann is not None:
        a = ann[name]
        rec.update(category=np.array(a['category']),
                   frame_labels=make_labels(reader.num_frames, a['intervals'], a['total']))
    return rec


def is_done(path: Path, with_q: bool) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as z:
        return not with_q or 'q' in z.files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    if args.video_dir is None:
        ap.error('--video-dir is required')
    names, videos, ann = select_videos(args)
    if not names:
        raise SystemExit('no videos selected')
    scores_dir = args.output_dir / 'scores'
    scores_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'config.json').write_text(json.dumps(
        {**cfg.protocol(), 'model_revision': args.model_revision, 'with_q': args.with_q, 'mode': 'offline',
         'max_windows': args.max_windows}, indent=1) + '\n')

    model = PHModel(args.device, args.hf_cache, args.with_q, revision=args.model_revision)
    print(f'{len(names)} videos -> {scores_dir} (probes: {", ".join(model.probes)})', flush=True)
    for i, name in enumerate(names, 1):
        out = scores_dir / f'{Path(name).stem}.npz'
        if is_done(out, args.with_q):
            print(f'[{i}/{len(names)}] SKIP {name}', flush=True)
            continue
        t0 = time.time()
        reader = ClipReader(videos[name])
        starts = reader.window_starts()[:args.max_windows]
        rows = process_windows(model, file_windows(reader, starts), pace='none', check=args.check)
        rec = video_record(name, reader, rows, args.with_q, ann)
        save_npz(out, **rec)
        msg = (f'[{i}/{len(names)}] {name}: {len(starts)} windows, mean PH {rec["score_PH"].mean():.3f}, '
               f'{(time.time() - t0) / len(starts):.2f} s/window')
        if 'check' in rec:
            msg += f' | check max |Δ| p, state, q vs reference {np.nanmax(rec["check"], 0).round(5).tolist()}'
        print(msg, flush=True)
    print(f'[DONE] {scores_dir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
