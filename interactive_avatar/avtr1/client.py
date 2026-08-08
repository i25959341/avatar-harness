from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import Avtr1Config


@dataclass(frozen=True)
class RenderedChunk:
    state: bytes
    frames_i420: list[bytes]
    width: int
    height: int


class Avtr1RendererClient:
    """Small client for AVTR-1's stateful five-frame renderer endpoint."""

    def __init__(self, config: Avtr1Config) -> None:
        self.config = config
        self._http = httpx.AsyncClient(
            base_url=config.renderer_url.rstrip("/") + "/",
            timeout=httpx.Timeout(5.0, connect=1.0),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def validate(self) -> None:
        health = await self._http.get("health")
        health.raise_for_status()
        response = await self._http.get("avatars")
        response.raise_for_status()
        catalog = response.json()
        if self.config.avatar_id not in catalog.get("avatars", []):
            raise RuntimeError(
                f"AVTR-1 avatar {self.config.avatar_id!r} is unavailable; "
                f"choose one of {catalog.get('avatars', [])}"
            )
        if self.config.background_id not in catalog.get("backgrounds", []):
            raise RuntimeError(
                f"AVTR-1 background {self.config.background_id!r} is unavailable; "
                f"choose one of {catalog.get('backgrounds', [])}"
            )

    async def render(
        self,
        speech_window: bytes,
        listen_window: bytes,
        state: bytes | None,
    ) -> RenderedChunk:
        current_bytes = self.config.current_samples * 2
        future_bytes = self.config.future_samples * 2
        expected = current_bytes + future_bytes
        if len(speech_window) != expected or len(listen_window) != expected:
            raise ValueError(f"AVTR-1 audio windows must contain exactly {expected} bytes")

        files: dict[str, tuple[str, bytes, str]] = {
            "current_chunk": ("speech-current.pcm", speech_window[:current_bytes], "audio/L16"),
            "future_chunk": ("speech-future.pcm", speech_window[current_bytes:], "audio/L16"),
            "current_chunk_listen": (
                "listen-current.pcm",
                listen_window[:current_bytes],
                "audio/L16",
            ),
            "future_chunk_listen": (
                "listen-future.pcm",
                listen_window[current_bytes:],
                "audio/L16",
            ),
            "state": ("state.safetensors", state or b"", "application/octet-stream"),
        }
        response = await self._http.post(
            "process-audio-v3",
            params={
                "avatar_id": self.config.avatar_id,
                "bg_id": self.config.background_id,
                "pixel_format": "yuv_i420",
                "cfg_self_audio": 2.0,
                "cfg_other_audio": 2.0,
                "cfg_kp": 3.0,
                "noise_alpha": 2.0,
                "noise_trunc_z": 1.2,
            },
            files=files,
        )
        response.raise_for_status()
        rendered = self._parse_response(response)
        self._validate_rendered_chunk(rendered)
        return rendered

    @staticmethod
    def _parse_response(response: httpx.Response) -> RenderedChunk:
        try:
            state_bytes = int(response.headers["X-State-Length-Bytes"])
            frame_bytes = int(response.headers["X-Frame-Length-Bytes"])
            frame_count = int(response.headers["X-Num-Frames"])
            width = int(response.headers["X-Frame-Width"])
            height = int(response.headers["X-Frame-Height"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(f"invalid AVTR-1 renderer headers: {exc}") from exc
        if min(state_bytes, frame_bytes, frame_count, width, height) < 0:
            raise RuntimeError("invalid AVTR-1 renderer headers: negative values are not allowed")
        if frame_bytes == 0 or frame_count == 0 or width == 0 or height == 0:
            raise RuntimeError("invalid AVTR-1 renderer headers: frame metadata must be positive")
        expected = state_bytes + frame_count * frame_bytes
        if len(response.content) != expected:
            raise RuntimeError(
                f"truncated AVTR-1 response: expected {expected} bytes, "
                f"received {len(response.content)}"
            )
        state = response.content[:state_bytes]
        frames = [
            response.content[state_bytes + index * frame_bytes : state_bytes + (index + 1) * frame_bytes]
            for index in range(frame_count)
        ]
        return RenderedChunk(state, frames, width, height)

    def _validate_rendered_chunk(self, rendered: RenderedChunk) -> None:
        if len(rendered.frames_i420) != self.config.chunk_frames:
            raise RuntimeError(
                "invalid AVTR-1 frame count: "
                f"expected {self.config.chunk_frames}, received {len(rendered.frames_i420)}"
            )
        if (rendered.width, rendered.height) != (self.config.width, self.config.height):
            raise RuntimeError(
                "invalid AVTR-1 frame dimensions: "
                f"expected {self.config.width}x{self.config.height}, "
                f"received {rendered.width}x{rendered.height}"
            )
        expected_bytes = self.config.width * self.config.height * 3 // 2
        invalid = [len(frame) for frame in rendered.frames_i420 if len(frame) != expected_bytes]
        if invalid:
            raise RuntimeError(
                "invalid AVTR-1 I420 frame length: "
                f"expected {expected_bytes} bytes, received {invalid[0]}"
            )
