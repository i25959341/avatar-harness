import httpx
import pytest

from interactive_avatar.avtr1.client import Avtr1RendererClient, RenderedChunk
from interactive_avatar.avtr1.config import Avtr1Config


def response(*, content: bytes, state: int = 3, frame: int = 4, count: int = 2) -> httpx.Response:
    return httpx.Response(
        200,
        headers={
            "X-State-Length-Bytes": str(state),
            "X-Frame-Length-Bytes": str(frame),
            "X-Num-Frames": str(count),
            "X-Frame-Width": "1280",
            "X-Frame-Height": "720",
        },
        content=content,
    )


def test_parse_renderer_response() -> None:
    result = Avtr1RendererClient._parse_response(response(content=b"abc11112222"))

    assert result.state == b"abc"
    assert result.frames_i420 == [b"1111", b"2222"]
    assert (result.width, result.height) == (1280, 720)


def test_parse_renderer_response_rejects_truncation() -> None:
    with pytest.raises(RuntimeError, match="truncated"):
        Avtr1RendererClient._parse_response(response(content=b"abc1111"))


def test_validate_renderer_response_geometry() -> None:
    client = object.__new__(Avtr1RendererClient)
    client.config = Avtr1Config(width=4, height=2, chunk_frames=2)
    valid_frame = bytes(12)

    client._validate_rendered_chunk(RenderedChunk(b"state", [valid_frame] * 2, 4, 2))

    with pytest.raises(RuntimeError, match="frame count"):
        client._validate_rendered_chunk(RenderedChunk(b"state", [valid_frame], 4, 2))
    with pytest.raises(RuntimeError, match="dimensions"):
        client._validate_rendered_chunk(RenderedChunk(b"state", [valid_frame] * 2, 8, 2))
    with pytest.raises(RuntimeError, match="I420 frame length"):
        client._validate_rendered_chunk(RenderedChunk(b"state", [bytes(11)] * 2, 4, 2))


def test_parse_renderer_response_rejects_nonpositive_metadata() -> None:
    with pytest.raises(RuntimeError, match="must be positive"):
        Avtr1RendererClient._parse_response(
            response(content=b"abc", frame=0, count=0)
        )
