from dataclasses import replace

import pytest

from interactive_avatar.interactive_avatarforcing.config import (
    InteractiveAvatarForcingConfig,
)


def test_interactive_avatarforcing_block_geometry() -> None:
    config = InteractiveAvatarForcingConfig()

    assert config.samples_per_frame == 640
    assert config.samples_per_block == 6_400
    assert config.bytes_per_block == 12_800
    assert config.history_bytes == 64_000
    assert config.block_seconds == 0.4


def test_interactive_avatarforcing_rejects_incompatible_geometry() -> None:
    config = replace(InteractiveAvatarForcingConfig(), block_frames=8)

    with pytest.raises(RuntimeError, match="requires 10-frame blocks"):
        config.validate()


def test_interactive_avatarforcing_requires_two_nfe() -> None:
    config = replace(InteractiveAvatarForcingConfig(), nfe=1)

    with pytest.raises(RuntimeError, match="NFE must be at least 2"):
        config.validate()


def test_interactive_avatarforcing_requires_rotary_capacity() -> None:
    config = replace(InteractiveAvatarForcingConfig(), rotary_max_frames=50)

    with pytest.raises(RuntimeError, match="rotary position capacity"):
        config.validate()


def test_interactive_avatarforcing_requires_camera_timeout() -> None:
    config = replace(InteractiveAvatarForcingConfig(), camera_stale_seconds=0)

    with pytest.raises(RuntimeError, match="camera stale timeout"):
        config.validate()
