import asyncio
import time
from types import SimpleNamespace

import numpy as np

from interactive_avatar.avtr1.session import Avtr1LiveAvatar
from interactive_avatar.interactive_avatarforcing.config import (
    InteractiveAvatarForcingConfig,
)
from interactive_avatar.interactive_avatarforcing.session import (
    InteractiveAvatarForcingLiveAvatar,
    PairedFrame,
)


class CancelTask:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class AudioSource:
    def __init__(self) -> None:
        self.clear_count = 0

    def clear_queue(self) -> None:
        self.clear_count += 1


def test_audio_retry_restores_only_the_current_epoch() -> None:
    for session_type in (Avtr1LiveAvatar, InteractiveAvatarForcingLiveAvatar):
        avatar = object.__new__(session_type)
        avatar._epoch = 3
        avatar._speech = bytearray(b"new-speech")
        avatar._listen = bytearray(b"new-listen")

        avatar._restore_consumed_audio(3, b"old-speech", b"old-listen")
        assert avatar._speech == b"old-speechnew-speech"
        assert avatar._listen == b"old-listennew-listen"

        avatar._restore_consumed_audio(2, b"stale", b"stale")
        assert avatar._speech == b"old-speechnew-speech"
        assert avatar._listen == b"old-listennew-listen"


def test_interactive_segment_start_invalidates_queued_idle() -> None:
    avatar = object.__new__(InteractiveAvatarForcingLiveAvatar)
    avatar.config = InteractiveAvatarForcingConfig()
    avatar._epoch = 4
    avatar._media = asyncio.Queue()
    avatar._media.put_nowait(
        PairedFrame(np.zeros((1, 1, 3), dtype=np.uint8), b"", avatar._epoch)
    )
    avatar._audio_source = AudioSource()
    avatar._speech = bytearray(b"old")
    avatar._segment_active = False
    avatar._segment_ended = True
    avatar._segment_samples = 10
    avatar._published_speech_samples = 10
    avatar._playback_started = True
    avatar._room = None

    asyncio.run(avatar._begin_segment())

    assert avatar._epoch == 5
    assert avatar._media.empty()
    assert avatar._audio_source.clear_count == 1
    assert avatar._speech == b""
    assert avatar._segment_active


def test_interactive_disconnect_releases_listener_ownership() -> None:
    avatar = object.__new__(InteractiveAvatarForcingLiveAvatar)
    tasks = [CancelTask(), CancelTask()]
    avatar._listener_identity = "listener"
    avatar._listener_tasks = tasks
    avatar._listen = bytearray(b"audio")
    avatar._camera = [np.zeros((1, 1, 3), dtype=np.uint8)]
    avatar._camera_updated_at = time.monotonic()

    avatar._on_participant_disconnected(SimpleNamespace(identity="listener"))

    assert all(task.cancelled for task in tasks)
    assert avatar._listener_identity is None
    assert avatar._listener_tasks == []
    assert not avatar._listen
    assert not avatar._camera
    assert avatar._camera_updated_at is None


def test_interactive_camera_frames_expire() -> None:
    avatar = object.__new__(InteractiveAvatarForcingLiveAvatar)
    avatar.config = InteractiveAvatarForcingConfig(camera_stale_seconds=0.5)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    avatar._camera = [frame]
    avatar._camera_updated_at = time.monotonic()
    assert avatar._fresh_camera_frames() == [frame]

    avatar._camera_updated_at = time.monotonic() - 1
    assert avatar._fresh_camera_frames() == []
