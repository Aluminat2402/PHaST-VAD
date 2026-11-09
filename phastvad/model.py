"""InternVL3.5 probes of PHaST-VAD: next-token read-outs only (no text generation).

One window = 8 frames (RGB uint8 [8, H, W, 3]) given to the model as a video ("Frame1: <image> ... Frame8: <image>"
from the official chat template), one 448 px tile per frame. Probes p and state (and optionally q) are asked on the
same clip. Read-outs use the log-softmax over the full vocabulary at the first answer position; every answer is the
log-sum-exp over its single-token variants (Yes / ' Yes' / yes / ' yes' / YES, No ..., A / ' A', ...), normalised
over the candidate set:
    p = P(No) / (P(Yes) + P(No)),   state = softmax over {A, B, C, D},   q = P(Yes) / (P(Yes) + P(No)).

Fast engine (one forward per window, as used for every reported number):
  * vision tower only on frames not seen before in this video / stream (consecutive windows share 4 of 8 frames;
    frame features are independent of each other) - frame-feature cache keyed by the source frame index;
  * the common prefix (chat header + 8 frames + "Watch the video clip.") is prefilled once;
  * the probe suffixes run as one right-padded batch on a copy of that KV cache.
`reference()` runs every probe as a separate full forward (slow; only for checks): fp16 accumulation order differs,
so read-outs agree with the fast engine only up to small fp16 differences.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import (ATTN, IMAGE_SIZE, MODEL_ID, MODEL_REVISION, NO_VARIANTS, PROMPTS, SAMPLE_FPS, STATE_LETTERS,
                     YES_VARIANTS)


def _sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


class PHModel:
    """Probe engine. Call reset() at the start of every video / stream, then model(clip, frame_indices) per window."""

    def __init__(self, device: str = 'cuda:0', cache_dir: Path | str | None = None, with_q: bool = False,
                 model_id: str = MODEL_ID, revision: str | None = MODEL_REVISION):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.probes = ('p', 'state', 'q') if with_q else ('p', 'state')
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, dtype=torch.float16, device_map={'': device},
            attn_implementation=ATTN).eval()
        self.device = next(self.model.parameters()).device
        tok = self.processor.tokenizer
        single = lambda variants: sorted({e[0] for e in (tok.encode(t, add_special_tokens=False) for t in variants)
                                          if len(e) == 1})
        self.yes_ids, self.no_ids = single(YES_VARIANTS), single(NO_VARIANTS)
        self.letter_ids = [single((l, ' ' + l)) for l in STATE_LETTERS]
        if not self.yes_ids or not self.no_ids or not all(self.letter_ids):
            raise RuntimeError('no single-token answer ids for this tokenizer')
        cfg = self.model.config
        self.image_token_id = cfg.image_token_id
        self.vision_kwargs = dict(vision_feature_layer=cfg.vision_feature_layer,
                                  vision_feature_select_strategy=cfg.vision_feature_select_strategy)
        self.pad_id = tok.pad_token_id
        self.video_size = self._video_size()
        self.reset()

    # ------------------------------------------------------------------------------------------------ helpers
    def _video_size(self) -> dict | None:
        """InternVL3.5-HF ships a 384 px video preprocessor config although its vision tower expects 448 px."""
        vp = getattr(self.processor, 'video_processor', None)
        size = getattr(getattr(self.model.config, 'vision_config', None), 'image_size', None)
        if vp is None or size is None or getattr(vp.size, 'height', None) is None:
            return None
        h, w = (size, size) if isinstance(size, int) else tuple(size)
        if (h, w) != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f'unexpected vision input size {h}x{w}')
        return None if (vp.size.height, vp.size.width) == (h, w) else {'height': int(h), 'width': int(w)}

    def _inputs(self, clip: np.ndarray, prompt: str):
        messages = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kw = {'videos': [clip], 'do_sample_frames': False,
              'video_metadata': [{'fps': SAMPLE_FPS, 'total_num_frames': len(clip), 'frames_indices': list(range(len(clip)))}]}
        if self.video_size is not None:
            kw['videos_kwargs'] = {'size': self.video_size}
        inputs = self.processor(text=[text], return_tensors='pt', **kw).to(self.device)
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.half()
        return inputs

    def _readout(self, log_probs: torch.Tensor, probe: str) -> np.ndarray:
        """Candidate log-probabilities: [log P(Yes), log P(No)] or [log P(A) .. log P(D)] (unnormalised)."""
        groups = [self.yes_ids, self.no_ids] if probe != 'state' else self.letter_ids
        return torch.stack([torch.logsumexp(log_probs[g], 0) for g in groups]).double().cpu().numpy()

    @staticmethod
    def _values(lp: dict) -> dict:
        out = {f'lp_{k}': v for k, v in lp.items()}
        out['p'] = float(1.0 / (1.0 + np.exp(lp['p'][0] - lp['p'][1])))          # P(No) / (P(Yes) + P(No))
        st = np.exp(lp['state'] - lp['state'].max())
        out['state'] = st / st.sum()
        if 'q' in lp:
            out['q'] = float(1.0 / (1.0 + np.exp(lp['q'][1] - lp['q'][0])))      # P(Yes) / (P(Yes) + P(No))
        return out

    # ---------------------------------------------------------------------------------------------- fast engine
    def reset(self) -> None:
        """Forget the frame-feature cache and the prompt token ids (new video / stream)."""
        self._frames: dict[int, torch.Tensor] = {}
        self._ids = None

    def _prompt_ids(self, clip):
        if self._ids is None:
            ids = [self._inputs(clip, PROMPTS[k])['input_ids'][0] for k in self.probes]
            n = min(len(x) for x in ids)
            differ = torch.stack([x[:n] for x in ids]).ne(ids[0][:n]).any(0)
            split = int(differ.nonzero()[0, 0]) if differ.any() else n - 1
            self._ids = (ids, split)
        return self._ids

    @torch.inference_mode()
    def __call__(self, clip: np.ndarray, frame_indices: list[int], timings: dict | None = None) -> dict:
        """Read-outs of one window. frame_indices = source frame index of every clip frame (frame-cache keys)."""
        ids, split = self._prompt_ids(clip)
        missing = [i for i, k in enumerate(frame_indices) if k not in self._frames]
        t0 = time.perf_counter()
        t1 = t2 = t0
        if missing:
            meta = [{'fps': SAMPLE_FPS, 'total_num_frames': len(missing), 'frames_indices': list(range(len(missing)))}]
            kw = {'size': self.video_size} if self.video_size is not None else {}
            pixels = self.processor.video_processor(videos=[clip[missing]], return_tensors='pt', do_sample_frames=False,
                                                    video_metadata=meta, **kw)['pixel_values_videos'].flatten(0, 1)
            t1 = time.perf_counter()
            feats = self.model.model.get_image_features(pixel_values=pixels.to(self.device, torch.float16),
                                                        **self.vision_kwargs).pooler_output
            for j, i in enumerate(missing):
                self._frames[frame_indices[i]] = feats[j]
            t2 = _sync()
        feats = torch.stack([self._frames[k] for k in frame_indices])
        keep = set(frame_indices)
        self._frames = {k: v for k, v in self._frames.items() if k in keep}
        # shared prefix: chat header + visual tokens of the 8 frames + common text
        prefix = ids[0][:split].to(self.device)
        embeds = self.model.get_input_embeddings()(prefix[None])
        mask = (prefix == self.image_token_id)[None, :, None]
        embeds = embeds.masked_scatter(mask, feats.reshape(-1, feats.shape[-1]).to(embeds.dtype))
        cache = self.model(inputs_embeds=embeds, use_cache=True, logits_to_keep=1).past_key_values
        t3 = _sync()
        # probe suffixes as one right-padded batch on the repeated prefix cache
        suffixes = [x[split:].to(self.device) for x in ids]
        n, length = len(suffixes), max(len(x) for x in suffixes)
        batch = torch.full((n, length), self.pad_id, dtype=torch.long, device=self.device)
        attention = torch.zeros((n, split + length), dtype=torch.long, device=self.device)
        for i, x in enumerate(suffixes):
            batch[i, :len(x)] = x
            attention[i, :split + len(x)] = 1
        cache.batch_repeat_interleave(n)
        hidden = self.model.model(input_ids=batch, attention_mask=attention, past_key_values=cache,
                                  use_cache=True).last_hidden_state
        last = hidden[torch.arange(n), torch.tensor([len(x) - 1 for x in suffixes])]
        log_probs = torch.log_softmax(self.model.lm_head(last).float(), dim=-1)
        if not torch.all(torch.isfinite(log_probs)):
            raise FloatingPointError('non-finite logits (fp16 overflow?)')
        out = self._values({k: self._readout(log_probs[i], k) for i, k in enumerate(self.probes)})
        t4 = time.perf_counter()
        if timings is not None:
            timings.update(preprocess=t1 - t0, vision=t2 - t1, prefill=t3 - t2, suffix=t4 - t3,
                           new_vision_frames=len(missing), tokens_prefix=split, tokens_suffix=[len(x) for x in suffixes])
        return out

    @torch.inference_mode()
    def reference(self, clip: np.ndarray) -> dict:
        """Same read-outs with one full, independent forward per probe (no caches). For checks only."""
        lp = {}
        for k in self.probes:
            logits = self.model(**self._inputs(clip, PROMPTS[k]), logits_to_keep=1).logits[0, -1].float()
            lp[k] = self._readout(torch.log_softmax(logits, -1), k)
        return self._values(lp)

    def warmup(self, n: int = 3, shape=(240, 320)) -> None:
        dummy = (np.random.default_rng(0).random((8, *shape, 3)) * 255).astype(np.uint8)
        for _ in range(n):
            self.reset()
            self(dummy, list(range(8)))
        self.reset()
        _sync()
