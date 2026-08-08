# IMTalker

IMTalker publishes 512x512 video at 25 FPS using adaptive 10-50 frame batches. During
generation it plays a cached moving idle sequence; completion and interruption use an
eight-frame latent transition back to that cache. The pinned upstream source is
Apache-2.0; review checkpoint and asset terms separately.

Validated on an RTX 5090 with PyTorch 2.7.1/CUDA 12.8. The model loads in about seven
seconds.

## Setup

Start from the common root environment in the [README](../../README.md#install), then
install IMTalker's additional runtime dependencies:

```bash
git submodule update --init --recursive avatar_models/imtalker

uv pip install --python .venv/bin/python \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python \
  numpy==1.26.4 "pillow>=10,<11" "markupsafe>=2,<3" librosa face-alignment \
  transformers huggingface-hub torchdiffeq gradio
uv pip install --python .venv/bin/python \
  livekit-agents==1.6.7 aiofiles==24.1.0 pydantic==2.12.3
```

The old upstream Gradio app and LiveKit require incompatible `aiofiles` versions. The
configuration above is for Avatar Harness live mode; use a separate environment for upstream
Gradio if a clean `pip check` is required.

Download checkpoints:

```bash
.venv/bin/hf download cbsjtu01/IMTalker \
  renderer.ckpt generator.ckpt \
  wav2vec2-base-960h/config.json \
  wav2vec2-base-960h/pytorch_model.bin \
  wav2vec2-base-960h/preprocessor_config.json \
  wav2vec2-base-960h/feature_extractor_config.json \
  --local-dir checkpoints
```

Generate the moving idle cache:

```bash
.venv/bin/python tools/generate_idle_cache.py \
  --source avatar_models/imtalker/assets/source_1.png \
  --driver assets/imtalker_idle_driver.mp4 \
  --output outputs/cache/imtalker_idle.pt \
  --preview outputs/imtalker/idle_preview.mp4
```

## Configuration

The live source image must match the identity used for the idle cache:

```dotenv
IMTALKER_SOURCE_IMAGE=avatar_models/imtalker/assets/source_1.png
IMTALKER_IDLE_CACHE=outputs/cache/imtalker_idle.pt
```

## Run

```bash
# Offline
.venv/bin/python examples/generate_imtalker_video.py

# LiveKit
python3 examples/livekit_imtalker_agent.py connect \
  --room imtalker-live --log-level info
python3 tools/create_livekit_join_url.py --room imtalker-live
```

## Constraints

- One active room per warmed GPU process.
- Initial speech waits for at least 400 ms of TTS PCM plus generation time.
- False-interruption playback resume is unsupported; normal barge-in clearing works.
- Upstream `app.py` initializes the model at import time, so Avatar Harness prewarms and reuses
  that instance.
