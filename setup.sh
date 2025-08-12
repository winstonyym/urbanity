#!/usr/bin/env bash
set -euo pipefail           # safer defaults
set -x                      # echo commands (remove if you prefer)

# ──────────────────────────────────────────────────────────────
PKG_VERSION="0.5.15"
CONDA_ENV="urbanity"

# ────────── argument parsing ──────────
BACKEND="pyg"

case "${1-}" in
  pyg|dgl|none) BACKEND="$1" ;;       # accept explicit choice
  "" ) ;;                              # empty → keep default (“pyg”)
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

# ────────── helper: create & configure conda env ──────────
setup_conda_env () {
    # 2. (Optional) remove old environment if present
    conda env remove -n "$CONDA_ENV" -y || true

    # 3. Create from environment.yml
    conda env create -n "$CONDA_ENV" -f environment.yml

    # 4. Activate
    conda activate "$CONDA_ENV"

    # Configure conda environment
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict

    # Install packages
    conda install mamba -c conda-forge -y
    mamba install geopandas -y
    python -m pip install uv
    
    uv pip install urbanity==$PKG_VERSION
    # 5. Uninstall pip version of networkit (if it exists)
    mamba install -c conda-forge -y networkit

    uv pip install setuptools==68
    uv pip install rasterio==1.4.0
    # mamba install -c conda-forge -y ipyleaflet 
    # mamba install -c conda-forge -y jupyter 
    conda activate "$CONDA_ENV"
    mamba install -c conda-forge -y pyarrow

    conda activate "$CONDA_ENV"
    mamba install -c conda-forge -y geemap

    conda activate "$CONDA_ENV"
    mamba install -c conda-forge -y osmium-tool

    conda activate "$CONDA_ENV"
    mamba install -c conda-forge -y pyrosm protobuf pyrobuf

    
    conda activate "$CONDA_ENV"
    # ── 4. install backend ─────────────────────────────────────────────
    echo "Installing backend '$BACKEND' (Device: $GPU_TYPE)"

    case "$BACKEND" in
    pyg)
        # ① Install the right PyTorch wheel first
        if [[ $GPU_TYPE == "cuda" ]]; then
            pip3 install torch torchvision 
            WHEEL_TAG="cu121"
        else                                       # CPU fallback
            pip3 install torch torchvision
            WHEEL_TAG="cpu"
        fi

        # ② Install the PyTorch‑Geometric stack that matches the wheel
        PYG_URL="https://data.pyg.org/whl/torch-2.6.0+${WHEEL_TAG}.html"
        uv pip install torch_geometric torch_scatter torch_sparse \
                    torch_cluster torch_spline_conv -f "${PYG_URL}"
        ;;

    dgl)
        if [[ $GPU_TYPE == "cuda" ]]; then
            # pick the CUDA build that matches your driver / toolkit
            uv pip install dgl-cu121            # change cuXYZ if needed
        else
            uv pip install dgl                  # CPU‑only or ROCm build
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