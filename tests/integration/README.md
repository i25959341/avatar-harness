# Integration Diagnostics

These scripts are executable diagnostics, not pytest tests. They write review media and
JSON reports under `outputs/` and may require CUDA, model checkpoints, FFmpeg, or
LiveKit credentials.

## IMTalker

```bash
.venv/bin/python tests/integration/imtalker_sync.py
```

This simulates streamed TTS input and renders the resulting paired media timeline.

## FlashHead

Run these inside the FlashHead environment, or invoke them with `python3`; scripts that
need that environment re-execute themselves automatically.

```bash
python3 tests/integration/flashhead_sync.py
python3 tests/integration/flashhead_interruption.py
python3 tests/integration/flashhead_interruption_options.py
python3 tests/integration/flashhead_idle_strategies.py
python3 tests/integration/flashhead_latent_bridge.py
python3 tests/integration/flashhead_livekit_loopback.py
```

`flashhead_sync.py` validates generated duration, frame rate, and motion.
`flashhead_interruption.py` renders idle/speech/interruption/resume behavior.
`flashhead_interruption_options.py` compares generated, hard-cut, pixel, hybrid, and
VAE-endpoint interruption handoffs from one controlled model state.
`flashhead_idle_strategies.py` compares candidate idle transitions.
`flashhead_latent_bridge.py` compares pixel and VAE-latent bridges.
`flashhead_livekit_loopback.py` publishes and receives a rendered sequence through a
real LiveKit room and reports arrival timing.

## Generated-Motion Backends

These benchmarks exercise the offline persistent-state paths used by the experimental
backends:

```bash
python3 benchmarks/interactive_avatarforcing_offline.py --nfe 10 4 2
python3 benchmarks/avtr1_continuous.py
```

Run each backend's setup tool before its benchmark.
