# FlashHead

SoulX-FlashHead Lite is TalkBox's recommended backend. It publishes 512x512 video at
25 FPS, plays a moving idle clip, and generates motion bridges around speech and
interruption. The pinned upstream source is Apache-2.0; review checkpoint and asset terms
separately.

Validated on an RTX 5090 with PyTorch 2.7.1/CUDA 12.8. A 24-frame, 960 ms chunk renders
in roughly 200-240 ms after warmup. Cold startup takes about 40 seconds.

## Setup

```bash
git submodule update --init --recursive third_party/SoulX-FlashHead

uv venv --python 3.10 third_party/SoulX-FlashHead/.venv
uv pip install --python third_party/SoulX-FlashHead/.venv/bin/python \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python third_party/SoulX-FlashHead/.venv/bin/python \
  -r third_party/SoulX-FlashHead/requirements.txt
uv pip install --python third_party/SoulX-FlashHead/.venv/bin/python \
  ninja flash_attn==2.8.0.post2 --no-build-isolation
uv pip install --python third_party/SoulX-FlashHead/.venv/bin/python \
  livekit-agents==1.6.7 aiofiles==24.1.0 pydantic==2.12.3 \
  "huggingface_hub[cli]"
```

Download the model and audio encoder:

```bash
third_party/SoulX-FlashHead/.venv/bin/hf download \
  Soul-AILab/SoulX-FlashHead-1_3B \
  --local-dir third_party/SoulX-FlashHead/models/SoulX-FlashHead-1_3B
third_party/SoulX-FlashHead/.venv/bin/hf download \
  facebook/wav2vec2-base-960h \
  --local-dir third_party/SoulX-FlashHead/models/wav2vec2-base-960h
```

FlashHead's optional MediaPipe dependency conflicts with LiveKit's Protobuf version. The
TalkBox runtime disables that face-crop path; `uv pip check` may report the upstream
metadata conflict.

## Configuration

Defaults use `avatar_models/imtalker/assets/source_1.png` and `assets/idle.mp4`. Override
them in `.env.local`:

```dotenv
FLASHHEAD_SOURCE_IMAGE=path/to/avatar.png
FLASHHEAD_IDLE_VIDEO=path/to/matching-idle.mp4
FLASHHEAD_INTERRUPTION_TRANSITION=vae
FLASHHEAD_INTERRUPTION_BRIDGE_FRAMES=4
```

The source and idle clip must show the same identity and framing. Interruption strategies
are `generated`, `vae`, `pixel`, and `hard`.

## Run

```bash
# Offline
python3 examples/generate_flashhead_video.py \
  --source avatar_models/imtalker/assets/source_1.png \
  --audio avatar_models/imtalker/assets/audio_1.wav \
  --output outputs/flashhead/demo.mp4

# LiveKit
python3 examples/livekit_flashhead_agent.py connect \
  --room flashhead-live --log-level info
python3 tools/create_livekit_join_url.py --room flashhead-live
```

The launcher enters FlashHead's environment automatically.

## Constraints

- One active room per warmed GPU process.
- Initial speech waits for one model chunk of TTS input.
- False-interruption playback resume is unsupported; normal barge-in clearing works.
- Generated media comparisons are available through the scripts in
  [tests/integration](../../tests/integration/README.md).
