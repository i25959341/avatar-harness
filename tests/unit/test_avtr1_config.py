from dataclasses import replace

import pytest

from interactive_avatar.avtr1 import Avtr1Config


def test_avtr1_stream_geometry() -> None:
    config = Avtr1Config()

    assert config.chunk_frames == 5
    assert config.chunk_seconds == 0.2
    assert config.samples_per_frame == 640
    assert config.bytes_per_frame == 1_280
    assert config.current_samples == 3_200
    assert config.future_samples == 3_280


def test_avtr1_sample_rate_must_divide_fps(tmp_path) -> None:
    config = replace(
        Avtr1Config(),
        avtr1_dir=tmp_path,
        sample_rate=16_001,
    )

    with pytest.raises(RuntimeError, match="sample rate must divide"):
        config.validate_setup()
