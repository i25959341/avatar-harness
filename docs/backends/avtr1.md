# AVTR-1

AVTR-1 generates 1280x720 video at 25 FPS from avatar speech, participant speech, and
silence. One autoregressive state covers speaking, listening, and idle motion, so no
cached idle clip is required. TalkBox publishes each five-frame I420 chunk with five
matching 40 ms audio slices.

This backend is experimental and noncommercial. AVTR-1 uses several licenses: its model
and scripts use the AVTR-1 Community License, renderer and streamer use PolyForm
Noncommercial 1.0.0, and bundled InsightFace models are restricted to noncommercial
research. Review the upstream license files before use.

## Setup

Requirements: Linux, an Ampere-or-newer NVIDIA GPU, CUDA 12.8, TensorRT 10.x, and access
to the gated [`avaturn-live/avtr-1`](https://huggingface.co/avaturn-live/avtr-1)
repository. Start from the common [TalkBox install](../../README.md#install), install
Pixi locally, authenticate, then run the setup tool:

```bash
git submodule update --init third_party/avtr-1
curl -fsSL https://pixi.sh/install.sh -o /tmp/pixi-install.sh
PIXI_HOME="$PWD/.pixi-local" PIXI_NO_PATH_UPDATE=1 sh /tmp/pixi-install.sh

cd third_party/avtr-1
../../.pixi-local/bin/pixi run -e renderer hf auth login
cd ../..
.venv/bin/python tools/setup_avtr1.py
```

Alternatively, put `HF_TOKEN` in `.env.local` after accepting access. Setup creates an
isolated Python 3.12 Pixi environment, downloads artifacts, and builds GPU-specific
TensorRT engines under ignored `third_party/avtr-1/artifacts/`.

## Configuration

```dotenv
AVTR1_RENDERER_URL=http://127.0.0.1:8000
AVTR1_AVATAR_ID=maria
AVTR1_BACKGROUND_ID=plain_white
```

Keep the default URL for the worker-managed local renderer. An already healthy endpoint
at the configured URL is reused.

## Run

```bash
# Offline speech sample
.venv/bin/python examples/generate_avtr1_video.py \
  --speech third_party/avtr-1/example/speaker_1.ogg \
  --output outputs/avtr1/speech.mp4

# LiveKit
python3 examples/livekit_avtr1_agent.py connect \
  --room avtr1-live --log-level info
python3 tools/create_livekit_join_url.py --room avtr1-live
```

Offline rendering also accepts `--listen` or `--duration`. The validated RTX 5090 build
renders a 200 ms chunk in approximately 111-115 ms.

## Constraints

- One stateful renderer stream per active avatar session.
- Local startup builds hardware-specific TensorRT engines and can take several minutes.
- LiveKit support remains experimental; validate barge-in behavior for your avatar and
  renderer build.
