#!/usr/bin/env bash
set -ex  # show commands and exit on error

PKG_VERSION=0.5.8
CONDA_ENV="urbanity"

# Function to set up Conda environment
setup_conda_env() {
    # 1. Source conda so 'conda' and 'conda activate' work
    source "$(conda info --base)/etc/profile.d/conda.sh"

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
    
    python -m pip install urbanity==$PKG_VERSION
    # 5. Uninstall pip version of networkit (if it exists)
    python -m pip uninstall -y networkit || true

    python -m pip install setuptools==68
    python -m pip install rasterio==1.4.0
    mamba install -c conda-forge -y ipyleaflet 
    mamba install -c conda-forge -y jupyter 
    mamba install -c conda-forge -y pyarrow
    mamba install -c conda-forge -y networkit
    mamba install -c conda-forge -y h5py
    mamba install -c conda-forge -y geemap
    
    # SVI workflow
    python -m pip install osmium
    python -m pip install torch torchvision
    python -m pip install transformers
    python -m pip install vt2geojson
    mamba install -c conda-forge -y opencv
}

# Detect the operating system
case "$OSTYPE" in
    linux-gnu*)
        echo "Detected Linux OS"
        sudo apt-get update
        sudo apt-get install -y gcc
        setup_conda_env
        sudo chown -R "$USER" .
        ;;
    darwin*)
        echo "Detected macOS"
        setup_conda_env
        ;;
    cygwin*|msys*)
        echo "Detected Windows (via Cygwin/MSYS)"
        source "/c/Users/$(whoami)/anaconda3/etc/profile.d/conda.sh"
        setup_conda_env
        ;;
    *)
        echo "Unknown OS type: $OSTYPE"
        exit 1
        ;;
esac

echo "All done! Environment set up successfully."

