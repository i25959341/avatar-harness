#!/usr/bin/env python3
"""Create a short-lived LiveKit Meet URL for a Talkbox room."""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
VENV_PYTHONS = (
    ROOT_DIR / ".venv" / "bin" / "python",
    ROOT_DIR / "third_party" / "SoulX-FlashHead" / ".venv" / "bin" / "python",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", default="flashhead-live")
    parser.add_argument("--identity", default="flashhead-viewer")
    parser.add_argument("--name", default="FlashHead Viewer")
    parser.add_argument("--ttl-minutes", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    if importlib.util.find_spec("livekit") is None:
        python = next((candidate for candidate in VENV_PYTHONS if candidate.is_file()), None)
        if python is None:
            print("error: no TalkBox environment with LiveKit was found", file=sys.stderr)
            return 1
        return subprocess.run(
            [str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
        ).returncode

    from livekit import api

    from interactive_avatar.environment import load_env_file

    args = parse_args()
    load_env_file(ROOT_DIR / ".env.local")
    required = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        print("error: missing " + ", ".join(missing), file=sys.stderr)
        return 1
    if args.ttl_minutes <= 0:
        print("error: --ttl-minutes must be positive", file=sys.stderr)
        return 1

    token = (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(args.identity)
        .with_name(args.name)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=args.room,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .with_ttl(datetime.timedelta(minutes=args.ttl_minutes))
        .to_jwt()
    )
    query = urllib.parse.urlencode({"liveKitUrl": os.environ["LIVEKIT_URL"], "token": token})
    print(f"https://meet.livekit.io/custom?{query}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
