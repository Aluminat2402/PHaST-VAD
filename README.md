# PHaST-VAD

**Probabilistic History-aware Semantic–Temporal Probing for Video Anomaly Detection.**
A lightweight, training-free video anomaly detector built on a frozen vision–language model
(InternVL3.5-2B). Each window is scored by reading next-token probabilities of two fixed questions; no text is
generated, no weights are trained, and prompts contain no dataset class names.

## Method

Every 2 s, the last 4 s of video (8 frames at 2 FPS) form a window. Two probes are asked on the same clip:

| Probe | Prompt (abridged; exact text in `phastvad/config.py`) | Read-out |
|---|---|---|
| normality | `Is this clip normal? Answer with only Yes or No.` | p = P(No) / (P(Yes) + P(No)) |
| temporal state | four options for the **last 2 s** (A active, B precursor, C aftermath, D ordinary) | a = P(A) / (P(A)+P(B)+P(C)+P(D)) |

Only the likelihood of option A is used; no state label is predicted. The **PH** scoring rule is

```
e_t  = max(a_t, a_{t-1}, a_{t-2}, a_{t-3})      # K = 3, reset at the start of every video / stream
h_t  = sqrt(a_t * e_t)
PH_t = sqrt(p_t * h_t)                         # all read-outs clipped to [1e-8, 1]
```

All settings (model revision, window, prompts, K, fusion) are fixed for every dataset; prompts are checked by SHA-256
at import. An optional third probe (`--with-q`, "Is an abnormal event happening in this clip?") adds the control scores
QH = sqrt(q·h) and QPH = (q·p·h)^(1/3).

**Causality.** A window score is emitted only after its window has been observed and processed, and depends only on
that window and the three preceding ones (window-level causality). For frame-level metrics, each window score is
assigned to the last 2 s of its window (mean over overlaps, the first ~2 s take the first available score), so a frame
score may use content up to ~1.5 s after that frame (~3.5 s at the start of a video). Streaming latency is measured
from window completion, not from event onset.

**Engine.** One forward pass per window: vision features are cached per source frame (consecutive windows share 4 of
8 frames), the shared prefix is prefilled once, and the probe suffixes run as one batch on its KV cache
(`phastvad/model.py`).

## Installation

```bash
pip install -r requirements.txt     # Python 3.12, CUDA 12.1; versions used for the reported results
```

The model is downloaded from HuggingFace at the pinned revision on first use (`--hf-cache` or `HF_HUB_CACHE` to reuse
a cache). One stream needs about 6 GB of GPU memory (5.62 GiB allocated in our runs, without q).
The model weights are subject to the [InternVL3.5 license](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF).

## Data

Annotation format (UCF-Crime test format), one video per line:
`<file.mp4> <category> <start> <end> [<start> <end> ...] [T=<n>]` with 0-based inclusive frame indices, `-1` for unused
slots, category `Normal` for normal videos, and optional `T=<n>` when intervals refer to n annotated frames.

| Dataset | Preparation |
|---|---|
| UCF-Crime | official `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` |
| XD-Violence | test annotation in the format above (`label_A` videos are `Normal`) |
| UBnormal | `python prepare_ubnormal.py <root> --output ubn_annotation.txt` (official test split in `splits/ubnormal/`) |
| MSAD | `python prepare_msad.py <root> --out <dir>` → protocol i annotation and protocol ii video list |

Do not add videos, frames or annotations of any dataset (in particular MSAD) to this repository.

## Usage

```bash
# offline scoring (resumable; --num-shards/--shard-index to split over GPUs)
CUDA_VISIBLE_DEVICES=0 python score_videos.py --video-dir <videos> --annotation <ann.txt> --output-dir out/ucf
python evaluate.py out/ucf/scores                      # --score PH (default) | p | a | h | pa | QH | QPH | q

# streaming replay on the wall clock (one stream per GPU), then the latency summary
CUDA_VISIBLE_DEVICES=0 python stream.py --video-dir <videos> --annotation <ann.txt> --output-dir out/rt_ucf
python stream.py --summary --output-dir out/rt_ucf

# live source: camera index, RTSP/HTTP URL, or a file played at its own frame rate
CUDA_VISIBLE_DEVICES=0 python stream.py --live rtsp://<host>/<stream> --output-dir out/live
```

`run.sh` wraps these commands with dataset paths from environment variables (see its header), e.g.
`bash run.sh smoke 0 <video.mp4>`, `bash run.sh offline ucf 0 1`, `bash run.sh eval msad_ii`.

From Python:

```python
from phastvad.model import PHModel
from phastvad.pipeline import file_windows, process_windows
from phastvad.video import ClipReader

model = PHModel('cuda:0')
reader = ClipReader('video.mp4')
rows = process_windows(model, file_windows(reader, reader.window_starts()), pace='none')
print([r['score_PH'] for r in rows])
```

Each video produces `scores/<video>.npz` with the window read-outs (`p`, `state`, optional `q`), window scores
(`score_PH`, …), frame scores, timings and, with `--annotation`, frame labels; `config.json` records the frozen
protocol.

## Results

InternVL3.5-2B, offline, frame-level metrics in %. Macro AUC/AP average over videos containing both normal and
abnormal frames.

| Protocol | Micro AUC | Micro AP | Macro AUC | Macro AP |
|---|---:|---:|---:|---:|
| UCF-Crime | 85.88 | 39.43 | 75.19 | 52.72 |
| XD-Violence | 93.58 | 80.11 | 78.88 | 70.23 |
| UBnormal | 71.15 | 76.77 | 68.06 | 80.11 |
| MSAD (protocol i) | 89.12 | 80.75 | 74.51 | 77.14 |
| MSAD (protocol ii) | 93.01 | 78.37 | 72.55 | 76.25 |

Streaming replay, one stream on one RTX 2080 Ti (fp16, without q), end-to-end latency over all windows on a shared
server; the deadline is the 2 s stride:

| Dataset | Windows | p50 (s) | p95 (s) | p99 (s) | > 2 s |
|---|---:|---:|---:|---:|---:|
| UCF-Crime | 18 378 | 0.62 | 1.04 | 1.29 | 0.01 % |
| XD-Violence | 48 363 | 0.63 | 1.10 | 1.34 | 0.05 % |
| UBnormal | 1 438 | 0.94 | 3.69 | 4.79 | 18.71 % |
| MSAD (protocol i) | 3 703 | 0.71 | 1.67 | 2.30 | 2.08 % |

Misses occur mostly in the first (cold) and video-end (terminal) windows, where decoding and seeking dominate; on
UBnormal (short, high-bitrate videos) they also reach regular windows. These numbers describe the measured hardware
and load only; they are not a latency guarantee for other settings.

## Repository

| Path | Content |
|---|---|
| `phastvad/config.py` | frozen protocol: model and revision, window, prompts (SHA-256 checked) |
| `phastvad/model.py` | `PHModel`: probes and the one-forward-per-window engine; `reference()` for checks |
| `phastvad/scoring.py` | PH and control scores (offline and online), window-to-frame mapping (numpy only) |
| `phastvad/metrics.py` | micro/macro AUC and AP, video AUC, normal-frame FPR@0.5 (numpy only) |
| `phastvad/video.py`, `phastvad/pipeline.py`, `phastvad/datasets.py` | clip reading, window loop and latency statistics, annotations |
| `score_videos.py`, `stream.py`, `evaluate.py`, `run.sh` | offline scoring, streaming replay / live input, evaluation, launcher |
| `prepare_ubnormal.py`, `prepare_msad.py`, `splits/` | annotation preparation |

## Citation

The paper is under review; citation details will be added after publication.
