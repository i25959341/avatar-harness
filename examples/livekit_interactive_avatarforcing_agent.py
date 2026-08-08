#!/usr/bin/env python3
"""Run TaekyungKi's interactive AvatarForcing model in LiveKit."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = ROOT_DIR / "third_party" / "InteractiveAvatarForcing"
UPSTREAM_PYTHON = UPSTREAM_DIR / ".venv" / "bin" / "python"
COMPAT_DIR = ROOT_DIR / "interactive_avatar" / "interactive_avatarforcing_compat"

if Path(sys.prefix).resolve() != UPSTREAM_PYTHON.parent.parent.resolve():
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(COMPAT_DIR), str(UPSTREAM_DIR), str(ROOT_DIR)]
    )
    os.execve(
        str(UPSTREAM_PYTHON),
        [str(UPSTREAM_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(UPSTREAM_DIR))

from livekit.agents import (  # noqa: E402
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobExecutorType,
    JobProcess,
    RoomInputOptions,
    cli,
    inference,
)

from interactive_avatar.environment import load_env_file  # noqa: E402
from interactive_avatar.interactive_avatarforcing import (  # noqa: E402
    InteractiveAvatarForcingConfig,
    InteractiveAvatarForcingRuntime,
)
from interactive_avatar.interactive_avatarforcing.session import (  # noqa: E402
    InteractiveAvatarForcingLiveAvatar,
)

logger = logging.getLogger("interactive-avatarforcing-agent")

load_env_file(ROOT_DIR / ".env.local")


def prewarm(proc: JobProcess) -> None:
    config = InteractiveAvatarForcingConfig.from_env()
    runtime = InteractiveAvatarForcingRuntime(config)
    load_seconds = runtime.load()
    logger.info("Interactive AvatarForcing loaded in %.2fs", load_seconds)
    warm = runtime.warmup()
    logger.info("Interactive AvatarForcing warmed in %.2fs", warm.total_seconds)
    proc.userdata["interactive_avatarforcing_config"] = config
    proc.userdata["interactive_avatarforcing_runtime"] = runtime


server = AgentServer(
    job_executor_type=JobExecutorType.THREAD,
    num_idle_processes=0,
    initialize_process_timeout=120.0,
    job_memory_warn_mb=10_000,
    setup_fnc=prewarm,
)


@server.rtc_session(agent_name="interactive-avatarforcing-avatar")
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    avatar = InteractiveAvatarForcingLiveAvatar(
        ctx.proc.userdata["interactive_avatarforcing_runtime"],
        ctx.proc.userdata["interactive_avatarforcing_config"],
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
        room_input_options=RoomInputOptions(close_on_disconnect=False),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(server)
