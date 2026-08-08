from interactive_avatar.adaptive_chunker import (
    AdaptiveAudioBuffer,
    AdaptiveChunkConfig,
    AdaptiveDeadlineTracker,
)


class EmptyOutputQueue:
    def qsize(self) -> int:
        return 0


def make_buffer() -> tuple[AdaptiveChunkConfig, AdaptiveAudioBuffer]:
    config = AdaptiveChunkConfig(
        min_chunk_frames=10,
        default_chunk_frames=10,
        max_chunk_frames=20,
    )
    tracker = AdaptiveDeadlineTracker(config, EmptyOutputQueue())
    return config, AdaptiveAudioBuffer(config, tracker)


def test_exact_boundary_emits_ordered_completion_marker() -> None:
    config, buffer = make_buffer()
    buffer.push_audio(bytes(config.bytes_per_frame * 20))

    first = buffer.try_get_chunk()
    second = buffer.try_get_chunk()
    assert first is not None and first.num_frames == 10 and not first.is_final_chunk
    assert second is not None and second.num_frames == 10 and not second.is_final_chunk

    buffer.mark_done()
    completion = buffer.try_get_chunk()
    assert completion is not None
    assert completion.num_frames == 0
    assert completion.is_final_chunk
    assert buffer.try_get_chunk() is None


def test_partial_final_frame_is_zero_padded() -> None:
    config, buffer = make_buffer()
    payload = b"\x01\x02" * 100
    buffer.push_audio(payload)
    buffer.mark_done()

    final = buffer.try_get_chunk()
    assert final is not None
    assert final.num_frames == 1
    assert final.is_final_chunk
    assert len(final.audio_bytes) == config.bytes_per_frame
    assert final.audio_bytes.startswith(payload)
    assert final.audio_bytes[len(payload) :] == bytes(config.bytes_per_frame - len(payload))
