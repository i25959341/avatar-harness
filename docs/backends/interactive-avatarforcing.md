# Interactive AvatarForcing

[TaekyungKi/AvatarForcing](https://github.com/TaekyungKi/AvatarForcing) generates
512x512 speech, listening, and idle motion in one causal model state. It conditions on
avatar audio, participant audio, and optional participant face motion. Missing, stale,
or invalid camera input falls back to zero-motion conditioning.

The backend is experimental and licensed CC BY-NC 4.0. Treat it as a noncommercial
evaluation unless you obtain separate permission.

## Setup

```bash
git submodule update --init third_party/InteractiveAvatarForcing
python3 tools/setup_interactive_avatarforcing.py
```

The setup tool creates an isolated PyTorch 2.7/CUDA 12.8 environment, downloads model
weights, and preprocesses the bundled benchmark sample. Use `--skip-install`,
`--skip-download`, or `--skip-sample-preprocess` to repeat only part of setup.

## Configuration

```dotenv
INTERACTIVE_AVATARFORCING_SOURCE_IMAGE=third_party/InteractiveAvatarForcing/data/rumi.jpg
INTERACTIVE_AVATARFORCING_NFE=4
INTERACTIVE_AVATARFORCING_SEED=25
```

NFE 4 is the live default. The runtime generates ten frames per 400 ms block, retains
its transformer KV cache, rolls back interrupted in-flight blocks, and extends the
upstream rotary-position table for long sessions.

## Run

```bash
# Offline NFE comparison
python3 benchmarks/interactive_avatarforcing_offline.py --nfe 10 4 2

# LiveKit
python3 examples/livekit_interactive_avatarforcing_agent.py connect \
  --room interactive-live --log-level info
python3 tools/create_livekit_join_url.py --room interactive-live
```

On the validated RTX 5090, NFE 4 generated a 15.87-second sample in 4.65 seconds with
about 9 GiB peak CUDA allocation. Persistent 400 ms blocks render in roughly 90-210 ms.

## Constraints

- One persistent model state per active room.
- Participant camera reactions require a detectable front-facing face.
- Generated motion is stochastic and should be evaluated with the intended portrait,
  voice, and camera conditions.
