"""Load legacy Wav2Vec weight-normalization checkpoints on modern Torch."""

import torch

parametrizations = getattr(torch.nn.utils, "parametrizations", None)
if parametrizations is not None and hasattr(parametrizations, "weight_norm"):
    # Transformers 4.30 selects this attribute when present, but the released
    # checkpoint uses legacy weight_g/weight_v state-dict keys.
    delattr(parametrizations, "weight_norm")
