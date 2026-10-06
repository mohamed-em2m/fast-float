# Expects `uv` on PATH and runs from the repo root.
# Uses .venv when present, otherwise the active environment.
_UV_PY=""
if [ -d ".venv" ]; then
    _UV_PY="--python .venv/bin/python"
fi

# shellcheck disable=SC2086
uv pip install $_UV_PY torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# shellcheck disable=SC2086
uv pip install $_UV_PY -r requirements.txt
