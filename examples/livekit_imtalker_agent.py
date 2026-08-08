#!/usr/bin/env python3
"""Run a LiveKit voice agent with the local IMTalker avatar."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
IMTALKER_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
if Path(sys.prefix).resolve() != IMTALKER_PYTHON.parent.parent.resolve():
    os.execv(
        str(IMTALKER_PYTHON),
        [str(IMTALKER_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "avatar_models" / "imtalker"))

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
from interactive_avatar.imtalker import IMTalkerConfig, IMTalkerLiveAvatar  # noqa: E402
from interactive_avatar.livekit_generator import preload_model  # noqa: E402

logger = logging.getLogger("imtalker-agent")


load_env_file(ROOT_DIR / ".env.local")


def prewarm(proc: JobProcess) -> None:
    config = IMTalkerConfig.from_env()
    config.validate()
    logger.info("Loading IMTalker model")
    agent = preload_model(device="cuda")
    proc.userdata["imtalker_agent"] = agent
    proc.userdata["imtalker_config"] = config
    logger.info("IMTalker model loaded")


server = AgentServer(
    num_idle_processes=0,
    initialize_process_timeout=120.0,
    job_memory_warn_mb=6_000,
    setup_fnc=prewarm,
)


@server.rtc_session(agent_name="imtalker-avatar")
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    avatar = IMTalkerLiveAvatar(
        ctx.proc.userdata["imtalker_config"],
        preloaded_agent=ctx.proc.userdata["imtalker_agent"],
    )
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
