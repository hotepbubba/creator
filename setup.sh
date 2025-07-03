#!/usr/bin/env bash
# Setup script for Creator Dashboard
# Usage: ./setup.sh [--with-models]

set -e

# install main requirements
pip install -r requirements.txt

# install submodule requirements
for req in Character-Generator/requirements.txt FLUX-LoRA-DLC/requirements.txt self-forcing/requirements.txt; do
    if [ -f "$req" ]; then
        pip install -r "$req"
    fi
done

# install flash-attn for Self-Forcing (skip CUDA build)
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation

# optionally download models for self-forcing
if [[ "$1" == "--with-models" ]]; then
    python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
import os
from pathlib import Path

cache_dir = os.environ.get("MODEL_CACHE", str(Path.home() / ".cache" / "creator"))

snapshot_download(
    repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    local_dir="self-forcing/wan_models/Wan2.1-T2V-1.3B",
    local_dir_use_symlinks=False,
    resume_download=True,
    repo_type="model",
    cache_dir=cache_dir,
)

hf_hub_download(
    repo_id="gdhe17/Self-Forcing",
    filename="checkpoints/self_forcing_dmd.pt",
    local_dir="self-forcing",
    local_dir_use_symlinks=False,
    cache_dir=cache_dir,
)
PY
fi
