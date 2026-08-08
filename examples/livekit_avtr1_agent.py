#!/usr/bin/env python3
"""Run a LiveKit voice agent with continuous AVTR-1 motion."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
AVATAR_HARNESS_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
if Path(sys.prefix).resolve() != AVATAR_HARNESS_PYTHON.parent.parent.resolve():
    os.execv(
        str(AVATAR_HARNESS_PYTHON),
        [str(AVATAR_HARNESS_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

sys.path.insert(0, str(ROOT_DIR))

from livekit.agents import (  # noqa: E402
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    cli,
    inference,
)

from interactive_avatar.avtr1 import (  # noqa: E402
    Avtr1Config,
    Avtr1LiveAvatar,
    Avtr1RendererClient,
    Avtr1RendererService,
)
from interactive_avatar.environment import load_env_file  # noqa: E402

logger = logging.getLogger("avtr1-agent")

load_env_file(ROOT_DIR / ".env.local")


def prewarm(proc: JobProcess) -> None:
    config = Avtr1Config.from_env()
    config.validate_setup()
    service = Avtr1RendererService(config)
    logger.info("Starting AVTR-1 renderer")
    service.start()
    proc.userdata["avtr1_config"] = config
    proc.userdata["avtr1_service"] = service


server = AgentServer(
    num_idle_processes=0,
    initialize_process_timeout=180.0,
    job_memory_warn_mb=10_000,
    setup_fnc=prewarm,
)


@server.rtc_session(agent_name="avtr1-avatar")
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    config = ctx.proc.userdata["avtr1_config"]
    avatar = Avtr1LiveAvatar(Avtr1RendererClient(config), config)
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
        room_input_options=RoomInputOptions(close_on_disconnect=False),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(server)
