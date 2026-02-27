# Optional Dependencies

This page expands on the optional extras listed in `pyproject.toml` and mentioned in
[the README installation section](../README.md#installation). Treat it as a quick reference for picking the
right extras for your workflow, explaining compatibility expectations and sample install commands.

## Extras overview

- `dev`: testing, linting, and development tooling for contributors.
- `faiss-gpu`: GPU-accelerated FAISS (CUDA required).
- `security`: `safety` and supporting packages for vulnerability scanning.

## Sample install commands

```bash
# Development workflow (tests, linting, and typing)
pip install -e .[dev]

# Security scan support (enables `safety`-backed commands)
pip install -e .[security]

# GPU FAISS support (requires CUDA toolkit and compatible wheels)
pip install -e .[faiss-gpu]

# Combine extras for a full local iteration loop
pip install -e .[dev,security,faiss-gpu]
```

## Compatibility notes

### `dev`

- Pulls in `pytest`, `ruff`, `mypy`, and other development dependencies.
- Ideal for contributors needing local tests, linting, typing, and coverage checks.
- Works on all supported platforms—standard Python wheels are published on PyPI.

### `faiss-gpu`

- Depends on FAISS GPU wheels that require a CUDA toolkit compatible with your OS
  and Python version (see NVIDIA compatibility matrix for guidance).
- If a GPU wheel is unavailable, fall back to the default CPU path (`faiss-cpu`) bundled
  with the main package.
- After installing, verify the GPU FAISS import works:

```bash
python -c "import faiss; print('faiss version', faiss.__version__)"
```

### `security`

- Adds `safety` and related packages so `mana-analyzer security-scan` (and other security commands)
  can run against your dependency lockfile.
- Recommended when you want to enforce dependency vulnerability checks in CI or local runs.

## Quick verification

Use these commands to confirm the extras are registered and active:

```bash
python -m pip show mana-analyzer
python -m pip show safety
python -m pip show pytest
```

If an extra appears missing, reinstall with the matching spec and ensure your virtual environment
is activated before rerunning commands.
