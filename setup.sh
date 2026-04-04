#!/usr/bin/env bash
set -euo pipefail           # safer defaults
set -x                      # echo commands (remove if you prefer)

# ──────────────────────────────────────────────────────────────
PKG_VERSION="0.6.0"
CONDA_ENV="urbanity"

# ────────── argument parsing ──────────
BACKEND="pyg"

case "${1-}" in
  pyg|dgl|none) BACKEND="$1" ;;
  "" ) ;;
  -h|--help)
      echo "Usage: $0 [pyg|dgl|none]   # default: pyg" ; exit 0 ;;
  * )
      echo "Unknown option '$1'. Use 'pyg', 'dgl', 'none', or omit." >&2 ; exit 1 ;;
esac
echo "→ Selected backend: $BACKEND"

# Check if GPU exists
GPU_TYPE="cpu"

# ---- NVIDIA / CUDA ----
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q .; then
        GPU_TYPE="cuda"
    fi
fi

# Before setup_conda_env function, install mamba in base if not available
if ! command -v mamba &> /dev/null; then
    echo "→ Installing mamba in base environment..."
    conda install -n base -c conda-forge mamba -y
fi

# ────────── helper: create & configure conda env ──────────
setup_conda_env () {
    # Remove old environment if present
    conda env remove -n "$CONDA_ENV" -y || true

    # Create from environment.yml
    conda env create -n "$CONDA_ENV" -f environment.yml

    # Add conda-forge and set strict priority
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict

    # Use conda run to execute everything inside the env
    conda run -n "$CONDA_ENV" mamba install geopandas -y
    conda run -n "$CONDA_ENV" python -m pip install uv

    conda run -n "$CONDA_ENV" uv pip install "urbanity==$PKG_VERSION"
    conda run -n "$CONDA_ENV" mamba install -c conda-forge networkit -y

    conda run -n "$CONDA_ENV" mamba install -c conda-forge -y \
    opencv pyarrow geemap osmium-tool pyrosm protobuf pyrobuf

    # pip packages in one go
    conda run -n "$CONDA_ENV" uv pip install \
    setuptools==68 rasterio==1.4.0 mapbox_vector_tile vt2geojson transformers mkdocs-material

    # ── 4. install backend ─────────────────────────────────────────────
    echo "Installing backend '$BACKEND' (Device: $GPU_TYPE)"

    case "$BACKEND" in
    pyg)
        # Install PyTorch (CPU or CUDA)
        if [[ $GPU_TYPE == "cuda" ]]; then
            conda run -n "$CONDA_ENV" pip3 install torch torchvision
            WHEEL_TAG="cu121"
        else
            conda run -n "$CONDA_ENV" pip3 install torch torchvision
            WHEEL_TAG="cpu"
        fi

        # Install PyTorch Geometric
        PYG_URL="https://data.pyg.org/whl/torch-2.6.0+${WHEEL_TAG}.html"
        conda run -n "$CONDA_ENV" uv pip install \
            torch_geometric torch_scatter torch_sparse \
            torch_cluster torch_spline_conv -f "$PYG_URL"
        ;;

    dgl)
        if [[ $GPU_TYPE == "cuda" ]]; then
            conda run -n "$CONDA_ENV" uv pip install dgl-cu121
        else
            conda run -n "$CONDA_ENV" uv pip install dgl
        fi
        ;;

    none)
        echo "No deep‑learning backend selected – skipping."
        ;;

    *)
        echo "Unknown backend '$BACKEND'. Use pyg, dgl, or none." >&2
        exit 1
        ;;
    esac
}

# ────────── OS detection (unchanged) ──────────
case "$OSTYPE" in
  linux-gnu*)
      sudo apt-get update
      sudo apt-get install -y gcc
      source "$(conda info --base)/etc/profile.d/conda.sh"
      setup_conda_env
      sudo chown -R "$USER" .
      ;;
  darwin*)
      source "$(conda info --base)/etc/profile.d/conda.sh"
      setup_conda_env ;;
  cygwin*|msys*)
      source "/c/Users/$(whoami)/anaconda3/etc/profile.d/conda.sh"
      setup_conda_env
      ;;
  *) echo "Unknown OS type: $OSTYPE" ; exit 1 ;;
esac

echo "🎉  Environment '$CONDA_ENV' ready (backend: $BACKEND)"
