"""Frozen protocol of PHaST-VAD. Nothing here is tuned per dataset; changing any value changes the method.

Prompts are reproduced byte for byte from the frozen campaign (their SHA-256 is checked at import). They never name a
dataset category (open-set setting) and carry no scene description / context.
"""
from __future__ import annotations

import hashlib

MODEL_ID = 'OpenGVLab/InternVL3_5-2B-HF'
MODEL_REVISION = '3f301ffcf3dcbb47893afae6650ea3e78d96fb6d'
DTYPE = 'float16'
ATTN = 'sdpa'

WINDOW_SEC = 4.0          # window length
STRIDE_SEC = 2.0          # a new window every 2 s; also the per-window processing deadline
CLIP_FRAMES = 8           # frames per window -> 2 fps, frame index = rint(t * source_fps)
SAMPLE_FPS = CLIP_FRAMES / WINDOW_SEC
IMAGE_SIZE = 448          # one 448 x 448 tile per frame (256 visual tokens)
MEMORY_K = 3              # e_t = max(a_t .. a_{t-K})
EPS = 1e-8                # every read-out is clipped to [EPS, 1]
DEADLINE_SEC = STRIDE_SEC

STATE_OPTIONS = ('A. An abnormal event is actively happening right now.\n'
                 'B. Things may be leading up to an abnormal event, but it has not started yet.\n'
                 'C. Aftermath: an abnormal event already happened earlier and only its consequences remain.\n'
                 'D. Ordinary, normal activity.\n')
PROMPTS = {
    # p = P(No) / (P(Yes) + P(No))   (anomaly score from the inverted question)
    'p': 'Watch the video clip. Is this clip normal? Answer with only Yes or No.',
    # a = P(A) over {A, B, C, D}     (state of the newest 2 s; Frame1-4 are context only)
    'state': ('Watch the video clip. It covers 4 seconds; the first half (Frame1 to Frame4) is only earlier context. '
              'Which option best describes the second half of the clip (Frame5 to Frame8)?\n' + STATE_OPTIONS +
              'Answer with only the letter.'),
    # optional: q = P(Yes) / (P(Yes) + P(No))
    'q': 'Watch the video clip. Is an abnormal event happening in this clip? Answer with only Yes or No.',
}
PROMPT_SHA256 = {
    'p': 'b269952ceab5759ba46bde239ebd5b2228b5fca0a072a9db6191060a6c3fec17',
    'state': '455ff2d3b7759f377da8bf30a2179e2e53d624b16eb3010cc51a8b704b92a21d',
    'q': 'ea4dd9c9febf0c617e303db4ea1b903e28587e5f93f9b2733b809389548cf805',
}
YES_VARIANTS = ('Yes', ' Yes', 'yes', ' yes', 'YES')
NO_VARIANTS = ('No', ' No', 'no', ' no', 'NO')
STATE_LETTERS = ('A', 'B', 'C', 'D')
STATE_NAMES = ('active', 'precursor', 'aftermath', 'ordinary')

for _k, _v in PROMPTS.items():
    if hashlib.sha256(_v.encode()).hexdigest() != PROMPT_SHA256[_k]:
        raise RuntimeError(f'prompt {_k!r} differs from the frozen protocol')


def protocol() -> dict:
    """JSON-serialisable description of the frozen protocol (written next to every output)."""
    return {'model': MODEL_ID, 'model_revision': MODEL_REVISION, 'dtype': DTYPE, 'attn': ATTN,
            'window_sec': WINDOW_SEC, 'stride_sec': STRIDE_SEC, 'frames': CLIP_FRAMES, 'sample_fps': SAMPLE_FPS,
            'frame_index': 'rint(t * source_fps)', 'image_size': IMAGE_SIZE, 'memory_k': MEMORY_K, 'eps': EPS,
            'assign': 'last 2 s of each window, mean over overlaps, linear interpolation elsewhere',
            'prompts': PROMPTS, 'prompt_sha256': PROMPT_SHA256}
