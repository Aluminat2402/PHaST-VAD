#!/usr/bin/env python3
"""Streaming PHaST-VAD on one GPU: every window is scored as soon as it is complete, with a latency log.

Sources
  --video-dir / --only-video   replay files as live streams on the wall clock, one stream at a time: window w arrives
                               at s_w + 4 s after the stream starts, deadline = arrival + 2 s (the protocol of the
                               reported runtime numbers). --fast disables the waiting (throughput mode: latency then
                               equals processing time and is NOT a real-time measurement).
  --live SOURCE                camera index, RTSP/HTTP URL, or a file played at its own frame rate (OpenCV); windows
                               s = 0, 2, 4, ... of the incoming stream until it ends or --max-windows.
Output (in --output-dir)
  windows.csv                  one line per window, appended as soon as the score exists: arrival / start / finish
                               (s since stream start), latency, window type, stage times, p, a, (q), scores
  scores/<stem>.npz            per video / stream: same fields as score_videos.py plus the timings (evaluate.py
                               works on it), peak VRAM
  latency_summary.json         p50 / p95 / p99 / max latency and % > 2 s by window type (cold / regular / terminal /
                               all); `--summary` recomputes it for an existing output directory.
Window types: cold = first window of a stream (opens the stream, decodes 8 frames), terminal = final file window
aligned to the video end (arrives < 2 s after the previous one), regular = all others. Deadline compliance holds only
for the measured hardware and load; latency is measured from window completion, not from event onset.

  CUDA_VISIBLE_DEVICES=0 python stream.py --video-dir V --only-video X.mp4 --output-dir out/rt
  CUDA_VISIBLE_DEVICES=0 python stream.py --live rtsp://camera/stream --output-dir out/live --with-q
  python stream.py --summary --output-dir out/rt
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from phastvad import config as cfg
from phastvad.pipeline import format_summary, latency_summary

FIELDS = ('stream', 'window', 'type', 'start_sec', 'arrival', 'start', 'finish', 'latency', 't_decode', 't_preprocess',
          't_vision', 't_prefill', 't_suffix', 't_score', 't_total', 'new_frames', 'new_vision_frames', 'p', 'a', 'q',
          'PH', 'QH', 'QPH')


def summarize(scores_dir: Path) -> dict:
    keys = ('window_starts_sec', 'arrival', 'start', 'finish', *(f't_{s}' for s in
                                                                 ('decode', 'preprocess', 'vision', 'prefill', 'suffix', 'score', 'total')))
    vids = []
    for f in sorted(scores_dir.glob('*.npz')):
        with np.load(f, allow_pickle=False) as z:
            if 'arrival' in z.files and len(z['arrival']):
                vids.append({k: z[k] for k in keys})
    if not vids:
        raise SystemExit(f'no stream outputs in {scores_dir}')
    return {'videos': len(vids), **latency_summary(vids)}


def main() -> int:
    import torch

    from score_videos import add_common_args, save_npz, select_videos, video_record
    from phastvad.model import PHModel
    from phastvad.pipeline import file_windows, live_windows, process_windows, to_arrays
    from phastvad.scoring import window_to_frames
    from phastvad.video import ClipReader, LiveCapture, window_types

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument('--live', default=None, help='live source: camera index, URL or file played in real time')
    ap.add_argument('--source-fps', type=float, default=None, help='live: override the nominal frame rate')
    ap.add_argument('--fast', action='store_true', help='file replay without waiting (throughput, not real time)')
    ap.add_argument('--summary', action='store_true', help='only recompute latency_summary.json of --output-dir')
    ap.add_argument('--quiet', action='store_true', help='do not print every window')
    args = ap.parse_args()
    scores_dir = args.output_dir / 'scores'
    if args.summary:
        s = summarize(scores_dir)
        (args.output_dir / 'latency_summary.json').write_text(json.dumps(s, indent=1) + '\n')
        print(f'{s["videos"]} streams, latency = finish - arrival (s)\n' + format_summary({k: v for k, v in s.items() if k != 'videos'}))
        return 0
    if (args.video_dir is None) == (args.live is None):
        ap.error('give exactly one of --video-dir or --live')
    pace = 'none' if args.fast or args.live else 'realtime'
    scores_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'config.json').write_text(json.dumps(
        {**cfg.protocol(), 'model_revision': args.model_revision, 'with_q': args.with_q,
         'mode': 'live' if args.live else ('replay-fast' if args.fast else 'replay-realtime'),
         'deadline_sec': cfg.DEADLINE_SEC, 'max_windows': args.max_windows}, indent=1) + '\n')

    t0 = time.perf_counter()
    model = PHModel(args.device, args.hf_cache, args.with_q, revision=args.model_revision)
    t_load = time.perf_counter() - t0
    model.warmup()
    print(f'{torch.cuda.get_device_name(0)}: model ready in {time.perf_counter() - t0:.1f} s (load {t_load:.1f} s), '
          f'probes {", ".join(model.probes)}, weights {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB', flush=True)

    log_path = args.output_dir / 'windows.csv'
    new_log = not log_path.is_file()
    log_file = log_path.open('a', newline='')
    log = csv.DictWriter(log_file, FIELDS, extrasaction='ignore')
    if new_log:
        log.writeheader()

    def emitter(stream: str, types):
        def emit(r):
            typ = types(r['window'])
            line = {'stream': stream, 'type': typ, 'latency': r['finish'] - r['arrival'], 'a': r['score_a'],
                    **{k: r[k] for k in FIELDS if k in r},
                    **{k: r[f'score_{k}'] for k in ('PH', 'QH', 'QPH') if f'score_{k}' in r}}
            log.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in line.items()})
            log_file.flush()
            if not args.quiet:
                extra = f' q={r["q"]:.3f} QH={r["score_QH"]:.3f}' if 'q' in r else ''
                print(f'  [{stream}] w{r["window"]:<4d} [{r["start_sec"]:7.1f}, {r["start_sec"] + cfg.WINDOW_SEC:7.1f}) s '
                      f'PH={r["score_PH"]:.3f} p={r["p"]:.3f} a={r["score_a"]:.3f}{extra} | latency '
                      f'{line["latency"]:.3f} s ({typ})', flush=True)
        return emit

    try:
        if args.live:
            cap = LiveCapture(args.live, args.source_fps)
            stream = Path(args.live).stem if Path(args.live).is_file() else 'live'
            torch.cuda.reset_peak_memory_stats()
            types = lambda w: 'cold' if w == 0 else 'regular'
            rows = process_windows(model, live_windows(cap, args.max_windows), 'none', emitter(stream, types), args.check)
            cap.close()
            dropped = sum(1 for r in rows if r.get('dropped'))
            arr = to_arrays(rows, args.with_q)
            n = int(arr['frame_indices'].max()) + 1 if len(arr['p']) else 0
            save_npz(scores_dir / f'{stream}.npz', filename=np.array(stream), source_fps=np.float64(cap.fps),
                     num_frames=np.int64(n), with_q=np.bool_(args.with_q), dropped_windows=np.int64(dropped),
                     peak_alloc_gib=np.float64(torch.cuda.max_memory_allocated() / 2 ** 30), **arr,
                     frame_scores=window_to_frames(n, cap.fps, arr['window_starts_sec'], arr['score_PH']).astype(np.float32))
            print(f'[DONE] {stream}: {len(arr["p"])} windows, {dropped} dropped', flush=True)
        else:
            names, videos, ann = select_videos(args)
            for i, name in enumerate(names, 1):
                stem = Path(name).stem
                t_open = time.perf_counter()
                reader = ClipReader(videos[name])
                t_open = time.perf_counter() - t_open
                starts = reader.window_starts()[:args.max_windows]
                wt = window_types(starts)
                torch.cuda.reset_peak_memory_stats()
                rows = process_windows(model, file_windows(reader, starts), pace, emitter(stem, lambda w: wt[w]), args.check)
                rec = video_record(name, reader, rows, args.with_q, ann)
                save_npz(scores_dir / f'{stem}.npz', **rec, open_s=np.float64(t_open), mode=np.array(pace),
                         peak_alloc_gib=np.float64(torch.cuda.max_memory_allocated() / 2 ** 30),
                         peak_reserved_gib=np.float64(torch.cuda.max_memory_reserved() / 2 ** 30))
                lat = rec['finish'] - rec['arrival']
                reg = lat[wt == 'regular']
                msg = (f'[{i}/{len(names)}] {name}: {len(lat)} windows | latency all p50 {np.median(lat):.3f} '
                       f'p99 {np.percentile(lat, 99):.3f} s, miss>2s {100 * np.mean(lat > cfg.DEADLINE_SEC):.2f}%')
                if len(reg):
                    msg += (f' | regular p50 {np.median(reg):.3f} p99 {np.percentile(reg, 99):.3f} s, '
                            f'miss>2s {100 * np.mean(reg > cfg.DEADLINE_SEC):.2f}%')
                msg += f' | peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB'
                if 'check' in rec:
                    msg += f' | check max |Δ| p, state, q {np.nanmax(rec["check"], 0).round(5).tolist()}'
                print(msg, flush=True)
    finally:
        log_file.close()
    s = summarize(scores_dir)
    (args.output_dir / 'latency_summary.json').write_text(json.dumps(s, indent=1) + '\n')
    print(f'\n{s["videos"]} streams in {scores_dir}, latency = finish - arrival (s)'
          + (' [fast replay: latency = processing time]' if pace == 'none' and not args.live else '') + '\n'
          + format_summary({k: v for k, v in s.items() if k != 'videos'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
