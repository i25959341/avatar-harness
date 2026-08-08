# TalkBox

TalkBox runs local talking-head models as LiveKit voice avatars. Every backend publishes
matching video frames and 16 kHz mono PCM from one 25 Hz clock, including interruption
handling and idle motion.

## Backends

| Backend | Resolution | Idle and listening | Status | License note |
| --- | --- | --- | --- | --- |
| [SoulX-FlashHead Lite](docs/backends/flashhead.md) | 512x512 | Moving clip with generated bridges | Recommended | Apache-2.0 source |
| [IMTalker](docs/backends/imtalker.md) | 512x512 | Cached motion latents | Supported | Apache-2.0 source |
| [Interactive AvatarForcing](docs/backends/interactive-avatarforcing.md) | 512x512 | Continuously generated | Experimental | CC BY-NC 4.0 |
| [AVTR-1](docs/backends/avtr1.md) | 1280x720 | Continuously generated | Experimental | Noncommercial components |

Upstream repositories are pinned as Git submodules. TalkBox does not redistribute model
checkpoints. Review each backend's license before use; the repository's Apache-2.0 license
does not replace upstream model or asset terms.

## Requirements

- Linux, Python 3.10, `uv`, FFmpeg, Git LFS
- NVIDIA GPU and a compatible CUDA driver
- A LiveKit project for live rooms

Validated locally on an RTX 5090 with CUDA 12.8, PyTorch 2.7.1, and
`livekit-agents==1.6.7`.

## Install

```bash
git clone --recurse-submodules https://github.com/i25959341/talkbox.git
cd talkbox
uv sync --extra dev
cp .env.example .env.local
```

Add `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` to `.env.local`, then
complete one backend setup:

- [FlashHead](docs/backends/flashhead.md#setup)
- [IMTalker](docs/backends/imtalker.md#setup)
- [Interactive AvatarForcing](docs/backends/interactive-avatarforcing.md#setup)
- [AVTR-1](docs/backends/avtr1.md#setup)

## Live Room

Start exactly one backend worker for a development room:

```bash
python3 examples/livekit_flashhead_agent.py connect \
  --room avatar-demo --log-level info
```

Replace `flashhead` with `imtalker`, `interactive_avatarforcing`, or `avtr1` to select a
different backend. Each launcher enters the backend's required environment automatically.
After the worker is warm, create a two-hour browser URL:

```bash
python3 tools/create_livekit_join_url.py --room avatar-demo
```

Use the worker's `start` command and LiveKit agent dispatch for deployment. The direct
`connect` command is intended for one-room development sessions.

## Offline Rendering

```bash
# FlashHead
python3 examples/generate_flashhead_video.py \
  --source avatar_models/imtalker/assets/source_1.png \
  --audio avatar_models/imtalker/assets/audio_1.wav

# IMTalker
.venv/bin/python examples/generate_imtalker_video.py

# Interactive AvatarForcing comparison
python3 benchmarks/interactive_avatarforcing_offline.py --nfe 10 4 2

# AVTR-1
.venv/bin/python examples/generate_avtr1_video.py \
  --speech third_party/avtr-1/example/speaker_1.ogg
```

Generated media and reports are written under `outputs/` and ignored by Git.

## Design

Backend adapters own model loading and mutable generation state. They consume TTS PCM,
publish paired audio/video directly through LiveKit sources, discard stale generations on
barge-in, and return to idle without using LiveKit's `AVSynchronizer`. See
[paired media publication](docs/design/av-sync.md).

One warmed model process serves one active room. Model state is not shared across rooms.

## Repository Layout

```text
interactive_avatar/  backend adapters and shared IMTalker primitives
examples/            live workers and offline entry points
benchmarks/          performance and quality comparisons
tools/               setup, cache, and room utilities
tests/unit/          fast automated tests
tests/integration/   explicit GPU, media, and LiveKit diagnostics
avatar_models/       IMTalker submodule
third_party/         other pinned model submodules
```

## Validate

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
uv lock --check
```

GPU and LiveKit diagnostics are listed in
[tests/integration/README.md](tests/integration/README.md).

## Adding a Backend

Keep upstream code in a pinned submodule and the TalkBox adapter in
`interactive_avatar/<backend>/`. A backend must validate its assets, prewarm once per
worker, accept 16 kHz mono PCM, publish paired media on one monotonic clock, invalidate
stale output after interruption, and document its license and concurrency limits.

## License

TalkBox code is Apache-2.0. Submodules, checkpoints, and sample assets retain their
upstream licenses and usage restrictions.
