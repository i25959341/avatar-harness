"""Live SoulX-FlashHead integration for Avatar Harness."""

from typing import Any

from .config import FlashHeadConfig

__all__ = ["FlashHeadConfig", "FlashHeadLiveAvatar", "FlashHeadRuntime"]


def __getattr__(name: str) -> Any:
    if name == "FlashHeadRuntime":
        from .runtime import FlashHeadRuntime

        return FlashHeadRuntime
    if name == "FlashHeadLiveAvatar":
        from .session import FlashHeadLiveAvatar

        return FlashHeadLiveAvatar
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
