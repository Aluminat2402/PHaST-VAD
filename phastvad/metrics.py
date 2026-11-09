"""Frame-level evaluation (numpy only). All metrics are returned in percent.

  mauc     micro ROC-AUC: all frames of all videos pooled (main metric)
  ap       micro average precision over the pooled frames (main metric for XD-Violence)
  macauc   macro ROC-AUC: mean per-video AUC over videos that contain both normal and abnormal frames
  macap    macro AP over the same videos
  vidauc   video-level AUC: video label (abnormal = category != Normal and >= 1 abnormal frame) vs mean frame score
  nfpr     normal-frame FPR@0.5: fraction of the frames of normal videos with score >= 0.5 (fixed, never tuned)
  offmac   padded macro AUC (diagnostic): per-video AUC after appending a (label 0, score 0) and a (label 1, score 1)
           pseudo-frame, so all-normal / all-abnormal videos count too
AUC averages the ranks of tied scores; AP is the step-wise sum over distinct thresholds (as sklearn).
"""
from __future__ import annotations

import numpy as np

KEYS = ('mauc', 'ap', 'macauc', 'macap', 'vidauc', 'nfpr', 'offmac')


def roc_auc(labels, scores) -> float | None:
    labels, scores = np.asarray(labels, np.uint8), np.asarray(scores, np.float64)
    pos = int(labels.sum())
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind='stable')
    s = scores[order]
    # average rank of every group of tied scores
    bounds = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    ranks = np.empty(len(s), np.float64)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        ranks[order[lo:hi]] = (lo + 1 + hi) / 2.0
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(labels, scores) -> float | None:
    labels, scores = np.asarray(labels, np.uint8), np.asarray(scores, np.float64)
    pos = int(labels.sum())
    if pos == 0 or pos == len(labels):
        return None
    order = np.argsort(-scores, kind='stable')
    y, s = labels[order], scores[order]
    tp = np.cumsum(y, dtype=np.int64)
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])      # last index of every distinct threshold
    tp_end = tp[ends]
    recall_gain = np.diff(np.r_[0, tp_end]) / pos
    precision = tp_end / (ends + 1.0)
    return float(np.sum(recall_gain * precision))


def is_normal_video(category: str, labels) -> bool:
    return category == 'Normal' or not np.asarray(labels).any()


def evaluate(frame_scores: list, labels: list, categories: list[str]) -> dict:
    """Metrics of one dataset. frame_scores / labels: one array per video (same length), categories: one per video."""
    if not frame_scores:
        raise ValueError('no videos')
    S = [np.asarray(s, np.float64) for s in frame_scores]
    Y = [np.asarray(y, np.uint8) for y in labels]
    normal = np.array([is_normal_video(c, y) for c, y in zip(categories, Y)])
    pooled_s, pooled_y = np.concatenate(S), np.concatenate(Y)
    per_auc = [v for v in (roc_auc(y, s) for s, y in zip(S, Y)) if v is not None]
    per_ap = [v for v in (average_precision(y, s) for s, y in zip(S, Y)) if v is not None]
    pct = lambda v: None if v is None else 100.0 * v
    m = {
        'mauc': pct(roc_auc(pooled_y, pooled_s)),
        'ap': pct(average_precision(pooled_y, pooled_s)),
        'macauc': pct(float(np.mean(per_auc))) if per_auc else None,
        'macap': pct(float(np.mean(per_ap))) if per_ap else None,
        'vidauc': pct(roc_auc((~normal).astype(np.uint8), np.array([s.mean() for s in S]))),
        'nfpr': pct(float(np.mean(np.concatenate([s for s, n in zip(S, normal) if n]) >= 0.5))) if normal.any() else None,
        'offmac': pct(float(np.mean([roc_auc(np.r_[0, y, 1], np.r_[0.0, s, 1.0]) for s, y in zip(S, Y)]))),
    }
    m.update(videos=len(S), normal_videos=int(normal.sum()), macro_videos=len(per_auc), frames=int(len(pooled_y)),
             abnormal_frames=int(pooled_y.sum()))
    return m
