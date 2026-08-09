from typing import Any

from .config import InteractiveAvatarForcingConfig

__all__ = [
    "InteractiveAvatarForcingConfig",
    "InteractiveAvatarForcingRuntime",
    "InteractiveBlockResult",
    "RuntimeSnapshot",
]


def __getattr__(name: str) -> Any:
    if name in {
        "InteractiveAvatarForcingRuntime",
        "InteractiveBlockResult",
        "RuntimeSnapshot",
    }:
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
