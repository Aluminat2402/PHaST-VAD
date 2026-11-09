"""PHaST-VAD: training-free, window-level causal video anomaly detection with a frozen VLM (two next-token probes).

PH is the scoring rule S = sqrt(p * h); PHaST-VAD is the framework around it.

Modules
  config    frozen protocol: model + revision, window / stride / frames, prompts (exact bytes)
  scoring   numpy only: online and offline PH (and the optional q scores), window -> frame assignment
  metrics   numpy only: micro / macro AUC and AP, video AUC, normal-frame FPR@0.5
  datasets  annotation parsing and frame labels
  video     decord clip reader, window grid, window types, live (OpenCV) capture
  model     InternVL3.5 probes with the fast engine (frame-feature cache, shared prefix, batched suffixes)
  pipeline  window loop shared by offline scoring and streaming replay (timings, latency statistics)

`scoring`, `metrics` and `datasets` do not import torch, so evaluation runs on CPU without the model stack.
"""
