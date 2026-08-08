#!/usr/bin/env python3
"""Publish and receive the FlashHead generated-bridge sequence through LiveKit.

This is a real LiveKit transport loopback. It publishes an existing rendered
sequence in real time; it does not run FlashHead inference or cancellation live.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
FLASHHEAD_PYTHON = ROOT_DIR / "third_party" / "SoulX-FlashHead" / ".venv" / "bin" / "python"
DEFAULT_INPUT = (
    ROOT_DIR / "outputs" / "flashhead" / "idle_strategy_comparison" / "04_generated_bridge.mp4"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "flashhead" / "livekit_bridge"
FPS = 25
SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
FRAME_DURATION = 1.0 / FPS
STATE_TIMELINE = (
    (0, "idle", 1),
    (38, "talking", 1),
    (100, "interrupted", 2),
    (125, "talking", 3),
    (200, "idle", 4),
)


def root_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def in_livekit_environment() -> bool:
    return Path(sys.prefix).resolve() == FLASHHEAD_PYTHON.parent.parent.resolve()


def reexec_in_livekit_environment() -> int:
    if not FLASHHEAD_PYTHON.is_file():
        raise RuntimeError(f"FlashHead Python not found: {FLASHHEAD_PYTHON}")
    return subprocess.run(
        [str(FLASHHEAD_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
    ).returncode


def probe_video(path: Path) -> tuple[int, int, int, float]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    video = metadata["streams"][0]
    numerator, denominator = video["r_frame_rate"].split("/", maxsplit=1)
    fps = float(numerator) / float(denominator)
    return int(video["width"]), int(video["height"]), int(video["nb_frames"]), fps


def decode_media(path: Path) -> tuple[list[bytes], bytes, int, int]:
    width, height, expected_frames, fps = probe_video(path)
    if not np.isclose(fps, FPS):
        raise RuntimeError(f"input must be {FPS} FPS, got {fps:g}")
    video_result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    frame_size = width * height * 3
    frames = [
        video_result.stdout[offset : offset + frame_size]
        for offset in range(0, len(video_result.stdout), frame_size)
    ]
    if len(frames) != expected_frames or any(len(frame) != frame_size for frame in frames):
        raise RuntimeError(f"decoded {len(frames)} video frames, expected {expected_frames}")

    audio_result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    required_audio_bytes = len(frames) * SAMPLES_PER_FRAME * 2
    audio = audio_result.stdout[:required_audio_bytes].ljust(required_audio_bytes, b"\0")
    return frames, audio, width, height


def token(api: Any, identity: str, room_name: str) -> str:
    return (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )


@dataclass
class Receiver:
    rtc: Any
    expected_video_frames: int
    expected_audio_samples: int
    expected_width: int
    expected_height: int
    video_frames: list[bytes] = field(default_factory=list)
    audio_bytes: bytearray = field(default_factory=bytearray)
    video_arrivals: list[float] = field(default_factory=list)
    video_timestamps_us: list[int] = field(default_factory=list)
    audio_arrivals: list[float] = field(default_factory=list)
    audio_samples: int = 0
    width: int = 0
    height: int = 0
    tasks: list[asyncio.Task[None]] = field(default_factory=list)
    tracks_ready: asyncio.Event = field(default_factory=asyncio.Event)
    video_done: asyncio.Event = field(default_factory=asyncio.Event)
    audio_done: asyncio.Event = field(default_factory=asyncio.Event)
    resolutions: dict[str, int] = field(default_factory=dict)
    recording: bool = False
    video_transport_arrivals: list[float] = field(default_factory=list)

    def attach(self, room: Any) -> None:
        @room.on("track_subscribed")
        def on_track_subscribed(track: Any, publication: Any, participant: Any) -> None:
            del publication, participant
            if track.kind == self.rtc.TrackKind.KIND_VIDEO:
                self.tasks.append(asyncio.create_task(self.consume_video(track)))
            elif track.kind == self.rtc.TrackKind.KIND_AUDIO:
                self.tasks.append(asyncio.create_task(self.consume_audio(track)))
            if len(self.tasks) >= 2:
                self.tracks_ready.set()

    async def consume_video(self, track: Any) -> None:
        stream = self.rtc.VideoStream(track, format=self.rtc.VideoBufferType.RGB24)
        try:
            async for event in stream:
                frame = event.frame
                resolution = f"{frame.width}x{frame.height}"
                self.resolutions[resolution] = self.resolutions.get(resolution, 0) + 1
                if not self.recording:
                    continue
                self.video_transport_arrivals.append(time.perf_counter())
                if frame.width != self.expected_width or frame.height != self.expected_height:
                    continue
                self.width = frame.width
                self.height = frame.height
                self.video_frames.append(bytes(frame.data))
                self.video_arrivals.append(time.perf_counter())
                self.video_timestamps_us.append(event.timestamp_us)
                if len(self.video_frames) >= self.expected_video_frames:
                    self.video_done.set()
        finally:
            await stream.aclose()

    async def consume_audio(self, track: Any) -> None:
        stream = self.rtc.AudioStream(
            track,
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            frame_size_ms=40,
        )
        try:
            async for event in stream:
                frame = event.frame
                if not self.recording:
                    continue
                remaining = self.expected_audio_samples - self.audio_samples
                if remaining <= 0:
                    self.audio_done.set()
                    continue
                samples = min(frame.samples_per_channel, remaining)
                self.audio_bytes.extend(bytes(frame.data)[: samples * 2])
                self.audio_samples += samples
                self.audio_arrivals.append(time.perf_counter())
                if self.audio_samples >= self.expected_audio_samples:
                    self.audio_done.set()
        finally:
            await stream.aclose()

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def begin_recording(self) -> None:
        self.video_frames.clear()
        self.audio_bytes.clear()
        self.video_arrivals.clear()
        self.video_transport_arrivals.clear()
        self.video_timestamps_us.clear()
        self.audio_arrivals.clear()
        self.audio_samples = 0
        self.video_done.clear()
        self.audio_done.clear()
        self.recording = True


def write_received_media(receiver: Receiver, output: Path) -> None:
    if not receiver.video_frames or not receiver.audio_bytes:
        raise RuntimeError("subscriber did not receive both audio and video")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".rgb", delete=False) as video_file:
        video_path = Path(video_file.name)
        for frame in receiver.video_frames:
            video_file.write(frame)
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as audio_file:
        audio_path = Path(audio_file.name)
        audio_file.write(receiver.audio_bytes)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-video_size",
                f"{receiver.width}x{receiver.height}",
                "-framerate",
                str(FPS),
                "-i",
                str(video_path),
                "-f",
                "s16le",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-i",
                str(audio_path),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "fast",
                "-c:a",
                "aac",
                "-shortest",
                str(output),
            ],
            check=True,
        )
    finally:
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


async def run_loopback(args: argparse.Namespace) -> dict[str, Any]:
    from livekit import api, rtc

    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if not os.environ.get(name):
            raise RuntimeError(f"missing {name}; expected it in .env.local or environment")

    frames, audio, width, height = decode_media(args.input)
    room_name = args.room or f"flashhead-bridge-{uuid.uuid4().hex[:10]}"
    publisher = rtc.Room()
    subscriber = rtc.Room()
    receiver = Receiver(
        rtc=rtc,
        expected_video_frames=len(frames),
        expected_audio_samples=len(frames) * SAMPLES_PER_FRAME,
        expected_width=width,
        expected_height=height,
    )
    receiver.attach(subscriber)

    await subscriber.connect(
        os.environ["LIVEKIT_URL"], token(api, "flashhead-loopback-receiver", room_name)
    )
    await publisher.connect(
        os.environ["LIVEKIT_URL"], token(api, "flashhead-loopback-publisher", room_name)
    )

    audio_source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=120)
    video_source = rtc.VideoSource(width, height)
    audio_track = rtc.LocalAudioTrack.create_audio_track("flashhead-audio", audio_source)
    video_track = rtc.LocalVideoTrack.create_video_track("flashhead-video", video_source)
    await publisher.local_participant.publish_track(
        audio_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    await publisher.local_participant.publish_track(
        video_track,
        rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=False,
            video_encoding=rtc.VideoEncoding(
                max_bitrate=3_000_000,
                max_framerate=FPS,
            ),
            degradation_preference=rtc.DegradationPreference.MAINTAIN_FRAMERATE_AND_RESOLUTION,
        ),
    )
    await asyncio.wait_for(receiver.tracks_ready.wait(), timeout=15)
    await asyncio.sleep(0.5)

    await publisher.local_participant.set_attributes(
        {"flashhead.state": "idle", "flashhead.generation_id": "0"}
    )
    preroll_started = time.perf_counter()
    for index in range(args.preroll_frames):
        target = preroll_started + index * FRAME_DURATION
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        await audio_source.capture_frame(
            rtc.AudioFrame(
                data=bytes(SAMPLES_PER_FRAME * 2),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=SAMPLES_PER_FRAME,
            )
        )
        video_source.capture_frame(
            rtc.VideoFrame(
                width=width,
                height=height,
                type=rtc.VideoBufferType.RGB24,
                data=frames[0],
            ),
            timestamp_us=time.time_ns() // 1000,
        )
    await audio_source.wait_for_playout()
    await asyncio.sleep(0.25)
    receiver.begin_recording()

    state_events: list[dict[str, Any]] = []
    state_by_frame = {frame: (state, generation) for frame, state, generation in STATE_TIMELINE}
    started = time.perf_counter()
    for index, video_data in enumerate(frames):
        target = started + index * FRAME_DURATION
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        if index in state_by_frame:
            state, generation = state_by_frame[index]
            event = {
                "frame": index,
                "media_time_seconds": index / FPS,
                "state": state,
                "generation_id": generation,
                "publish_wall_time_seconds": time.perf_counter() - started,
            }
            state_events.append(event)
            await publisher.local_participant.set_attributes(
                {
                    "flashhead.state": state,
                    "flashhead.generation_id": str(generation),
                }
            )

        audio_start = index * SAMPLES_PER_FRAME * 2
        audio_data = audio[audio_start : audio_start + SAMPLES_PER_FRAME * 2]
        await audio_source.capture_frame(
            rtc.AudioFrame(
                data=audio_data,
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=SAMPLES_PER_FRAME,
            )
        )
        video_source.capture_frame(
            rtc.VideoFrame(
                width=width,
                height=height,
                type=rtc.VideoBufferType.RGB24,
                data=video_data,
            ),
            timestamp_us=time.time_ns() // 1000,
        )

    await audio_source.wait_for_playout()
    publish_elapsed = time.perf_counter() - started
    await asyncio.sleep(2.0)
    await publisher.disconnect()
    await asyncio.sleep(0.5)
    await receiver.stop()
    await subscriber.disconnect()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    received_path = output_dir / "flashhead_bridge_livekit_received.mp4"
    write_received_media(receiver, received_path)

    first_av_skew_ms: float | None = None
    if receiver.video_transport_arrivals and receiver.audio_arrivals:
        first_av_skew_ms = (
            receiver.video_transport_arrivals[0] - receiver.audio_arrivals[0]
        ) * 1000
    received_video_duration = len(receiver.video_frames) / FPS
    received_audio_duration = receiver.audio_samples / SAMPLE_RATE
    video_delivery_duration = (
        receiver.video_arrivals[-1] - receiver.video_arrivals[0]
        if len(receiver.video_arrivals) > 1
        else 0.0
    )
    expected_duration = len(frames) / FPS
    checks = {
        "video_frames_within_one": abs(len(receiver.video_frames) - len(frames)) <= 1,
        "audio_duration_within_80ms": abs(received_audio_duration - expected_duration) <= 0.08,
        "first_av_arrival_within_120ms": first_av_skew_ms is not None
        and abs(first_av_skew_ms) <= 120,
        "video_delivery_within_250ms": abs(
            video_delivery_duration - (expected_duration - FRAME_DURATION)
        )
        <= 0.25,
    }
    report: dict[str, Any] = {
        "scope": "real LiveKit publisher/subscriber loopback using pre-rendered FlashHead media",
        "room": room_name,
        "input": str(args.input),
        "received_output": str(received_path),
        "expected_video_frames": len(frames),
        "received_video_frames": len(receiver.video_frames),
        "expected_duration_seconds": expected_duration,
        "received_video_duration_seconds": received_video_duration,
        "received_audio_duration_seconds": received_audio_duration,
        "publish_elapsed_seconds": publish_elapsed,
        "video_delivery_duration_seconds": video_delivery_duration,
        "first_av_arrival_skew_ms": first_av_skew_ms,
        "first_video_rtp_timestamp_us": (
            receiver.video_timestamps_us[0] if receiver.video_timestamps_us else None
        ),
        "received_resolutions": receiver.resolutions,
        "idle_preroll_frames": args.preroll_frames,
        "state_events": state_events,
        "checks": checks,
        "passed": all(checks.values()),
        "not_measured": [
            "live FlashHead inference",
            "dynamic generation cancellation",
            "runtime queue flush",
        ],
    }
    report_path = output_dir / "flashhead_bridge_livekit_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FlashHead LiveKit bridge loopback.")
    parser.add_argument("--input", type=root_path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=root_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--room", help="room name; defaults to a unique temporary name")
    parser.add_argument(
        "--preroll-frames",
        type=int,
        default=25,
        help="idle frames published before measurement to warm the video track",
    )
    return parser.parse_args()


def main() -> int:
    if not in_livekit_environment():
        try:
            return reexec_in_livekit_environment()
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    load_env_file(ROOT_DIR / ".env.local")
    args = parse_args()
    try:
        report = asyncio.run(run_loopback(args))
    except (RuntimeError, subprocess.CalledProcessError, TimeoutError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
