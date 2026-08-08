#!/usr/bin/env python3
"""Run a LiveKit voice agent with a local SoulX-FlashHead avatar."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
FLASHHEAD_PYTHON = ROOT_DIR / "third_party" / "SoulX-FlashHead" / ".venv" / "bin" / "python"
if Path(sys.prefix).resolve() != (FLASHHEAD_PYTHON.parent.parent).resolve():
    os.execv(
        str(FLASHHEAD_PYTHON), [str(FLASHHEAD_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
    )

sys.path.insert(0, str(ROOT_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from livekit.agents import (  # noqa: E402
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
)

from interactive_avatar.environment import load_env_file  # noqa: E402
from interactive_avatar.flashhead import (  # noqa: E402
    FlashHeadConfig,
    FlashHeadLiveAvatar,
    FlashHeadRuntime,
)

logger = logging.getLogger("flashhead-agent")


load_env_file(ROOT_DIR / ".env.local")


def vae_warmup_windows(
    config: FlashHeadConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    capture = cv2.VideoCapture(str(config.idle_video))
    frames = []
    try:
        while len(frames) < config.motion_frames + config.interruption_bridge_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(
                frame,
                (config.width, config.height),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"could not decode VAE warm-up frames from {config.idle_video}")
    while len(frames) < config.motion_frames + config.interruption_bridge_frames:
        frames.append(frames[-1].copy())
    return frames[: config.motion_frames], frames[-config.motion_frames :]


def prewarm(proc: JobProcess) -> None:
    config = FlashHeadConfig.from_env()
    runtime = FlashHeadRuntime(config)
    load_seconds = runtime.load()
    logger.info("FlashHead loaded in %.1fs; compiling warm path", load_seconds)
    warmup_seconds = runtime.warmup()
    logger.info("FlashHead warm-up completed in %.1fs", warmup_seconds)
    if config.interruption_transition == "vae":
        source, target = vae_warmup_windows(config)
        started = time.perf_counter()
        runtime.generate_vae_transition(
            source,
            target,
            config.interruption_bridge_frames,
        )
        logger.info(
            "FlashHead VAE interruption path warmed in %.1fs",
            time.perf_counter() - started,
        )
    proc.userdata["flashhead_runtime"] = runtime
    proc.userdata["flashhead_config"] = config


server = AgentServer(
    # A spare process would load a second CUDA model while one room is active.
    num_idle_processes=0,
    initialize_process_timeout=120.0,
    job_memory_warn_mb=6_000,
    setup_fnc=prewarm,
)


@server.rtc_session(agent_name="flashhead-avatar")
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    runtime = ctx.proc.userdata["flashhead_runtime"]
    config = ctx.proc.userdata["flashhead_config"]
    avatar = FlashHeadLiveAvatar(runtime, config)
    await avatar.start(ctx.room)
    ctx.add_shutdown_callback(avatar.aclose)

    session = AgentSession(
        vad=inference.VAD(),
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS(
            "cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
    )
    session.output.replace_audio_tail(avatar.audio_output)
    await session.start(
        agent=Agent(
            instructions=(
                "You are a concise, conversational assistant represented by a live visual "
                "avatar. Respond naturally and keep ordinary answers brief."
            )
        ),
        room=ctx.room,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(server)
