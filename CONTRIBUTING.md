# Contributing

## Development Setup

Clone with submodules and follow the setup guide for the backend you intend to change.
Install the local package and development tools in that backend's environment:

```bash
uv pip install --python .venv/bin/python -e '.[dev]'
```

Do not commit model checkpoints, generated media, environment files, or local virtual
environments.

## Checks

Before opening a pull request, run:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

Changes to model generation, transition behavior, timing, or publication must also run
the relevant command from `tests/integration/README.md`. Include the generated JSON
metrics in the pull request description, but do not commit files from `outputs/`.

## Backend Contract

New backends belong in `interactive_avatar/<backend>/`; keep upstream model source in a
pinned submodule. Validate required assets before loading CUDA, avoid blocking the
asyncio event loop, use a single clock for paired audio/video publication, and make
interruption invalidate queued and in-flight output.
