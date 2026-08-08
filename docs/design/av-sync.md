# Paired Audio/Video Publication

## Decision

TalkBox backends publish audio and video directly through `rtc.AudioSource` and
`rtc.VideoSource`. They do not send generated media through LiveKit's `AvatarRunner` or
`AVSynchronizer`.

Each publication iteration emits:

- one video frame at 25 FPS;
- exactly 640 samples of 16 kHz mono signed 16-bit PCM;
- both items from the same backend generation epoch.

This gives every frame a matching 40 ms audio slice and keeps interruption handling in
the adapter that owns the model state.

## Why

TalkBox models generate batches faster than real time, then publish them at a steady
rate. `AVSynchronizer` independently queues and paces video while forwarding audio
through a different path. With bursty model output, that architecture previously caused
audio to lead video even when locally rendered media was synchronized.

Direct paired publication removes the second scheduler. The backend owns buffering,
pacing, and stale-output invalidation.

## Invariants

Every live backend must:

1. Accept 16 kHz mono PCM from `QueueAudioOutput`.
2. Divide audio into one 640-sample slice per generated frame.
3. Publish both streams from one monotonic 25 Hz loop.
4. Tag generated work with a generation epoch or equivalent invalidation marker.
5. Advance that marker and drop stale work when playback is interrupted.
6. Clear pending RTC audio on interruption.
7. Notify LiveKit when playback starts and finishes.

Backend-specific idle behavior may differ, but idle frames still carry matching silent
audio slices.

## Validation

Fast lifecycle and buffer tests run with:

```bash
.venv/bin/python -m pytest
```

GPU and LiveKit diagnostics are listed in
[tests/integration/README.md](../../tests/integration/README.md). Generated review media
is written under `outputs/` and is not versioned.
