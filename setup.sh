#!/usr/bin/env bash
set -euo pipefail           # safer defaults
set -x                      # echo commands (remove if you prefer)

# ──────────────────────────────────────────────────────────────
PKG_VERSION="0.5.10"
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
    uv pip uninstall -y networkit || true

    uv pip install setuptools==68
    uv pip install rasterio==1.4.0
    mamba install -c conda-forge -y ipyleaflet 
    mamba install -c conda-forge -y jupyter 
    mamba install -c conda-forge -y pyarrow
    mamba install -c conda-forge -y networkit
    mamba install -c conda-forge -y geemap
    mamba install -c conda-forge -y osmium-tool
    
    # ── 4. install backend ─────────────────────────────────────────────
    echo "Installing backend '$BACKEND' (Device: $GPU_TYPE)"

    case "$BACKEND" in
    pyg)
        # ① Install the right PyTorch wheel first
        if [[ $GPU_TYPE == "cuda" ]]; then
            uv pip install torch torchvision
            WHEEL_TAG="cu121"
        else                                       # CPU fallback
            uv pip install torch torchvision
            WHEEL_TAG="cpu"
        fi

        # ② Install the PyTorch‑Geometric stack that matches the wheel
        PYG_URL="https://data.pyg.org/whl/torch-2.4.0+${WHEEL_TAG}.html"
        uv pip install torch_geometric torch_scatter torch_sparse \
                    torch_cluster torch_spline_conv -f "${PYG_URL}"
        ;;

    dgl)
        if [[ $GPU_TYPE == "cuda" ]]; then
            # pick the CUDA build that matches your driver / toolkit
            uv install dgl-cu121            # change cuXYZ if needed
        else
            uv install dgl                  # CPU‑only or ROCm build
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

  # 5. misc (common) extras
  uv pip install transformers vt2geojson
  mamba install -y opencv -c conda-forge
}

# ────────── OS detection ──────────
detect_ostype() {
  # 1. Prefer Bash-provided $OSTYPE …
  if [[ -n "${OSTYPE-}" ]]; then
      printf '%s\n' "${OSTYPE,,}"   # ,, → lower-case
      return
  fi

  # 2. …otherwise fall back to uname(1)
  uname -s | tr '[:upper:]' '[:lower:]'
}

OS_ID=$(detect_ostype)

# Helper: source the *right* conda activation script everywhere
source_conda() {
  local base
  if ! base=$(conda info --base 2>/dev/null); then
      echo "conda not found – please install Miniconda/Anaconda first." >&2
      exit 1
  fi
  source "$base/etc/profile.d/conda.sh"
}

case "$OS_ID" in
  linux-gnu*|linux*)
      # Detect WSL (optional)
      if grep -qi microsoft /proc/version 2>/dev/null ; then
          echo "→ Running under WSL"
      fi
      sudo apt-get update
      sudo apt-get install -y gcc
      source_conda
      setup_conda_env
      sudo chown -R "$USER" .
      ;;

  darwin*)
      source_conda
      setup_conda_env
      ;;

  msys*|mingw*|cygwin*|win32*)
      # Git-Bash, MSYS2 or Cygwin
      source_conda
      setup_conda_env
      ;;

  *)
      echo "Unknown OS type: $OS_ID" >&2
      exit 1
      ;;
esac

echo "🎉  Environment '$CONDA_ENV' ready (backend: $BACKEND)"