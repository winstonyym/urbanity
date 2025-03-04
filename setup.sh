#!/usr/bin/env bash
set -ex  # show commands and exit on error

# Detect the operating system using OSTYPE:
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux OS"
    # --------------------------------------------------
    # Linux Installation
    sudo apt-get update
    sudo apt-get install gcc

    # 1. Source conda so 'conda' and 'conda activate' work
    source "$(conda info --base)/etc/profile.d/conda.sh"

    # 2. (Optional) remove old environment if present
    conda env remove -n urbanity -y || true

    # 3. Create from environment.yml
    conda env create -n urbanity -f environment.yml

    # 4. Activate
    conda activate urbanity

    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict

    conda install mamba -c conda-forge -y

    mamba install geopandas -y
    python -m pip install urbanity==0.5.8
    # 5. Uninstall pip version of networkit (if it exists)
    python -m pip uninstall -y networkit || true

    python -m pip install setuptools==68
    python -m pip install rasterio==1.4.0
    mamba install -c conda-forge -y ipyleaflet 
    mamba install -c conda-forge -y jupyter 
    mamba install -c conda-forge -y pyarrow
    mamba install -c conda-forge -y networkit
    mamba install -c conda-forge -y h5py

    # Set permission for saving
    sudo chown -R $USER .

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    # --------------------------------------------------
    # macOS Installation
    # 1. Source conda so 'conda' and 'conda activate' work
    source "$(conda info --base)/etc/profile.d/conda.sh"

    # 2. (Optional) remove old environment if present
    conda env remove -n urbanity -y || true

    # 3. Create from environment.yml
    conda env create -n urbanity -f environment.yml

    # 4. Activate
    conda activate urbanity

    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict

    conda install mamba -c conda-forge -y

    mamba install geopandas -y
    python -m pip install urbanity==0.5.8
    # 5. Uninstall pip version of networkit (if it exists)
    python -m pip uninstall -y networkit || true

    python -m pip install setuptools==68
    python -m pip install rasterio==1.4.0
    mamba install -c conda-forge -y ipyleaflet 
    mamba install -c conda-forge -y jupyter 
    mamba install -c conda-forge -y pyarrow
    mamba install -c conda-forge -y networkit
    mamba install -c conda-forge -y h5py

elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
    echo "Detected Windows (via Cygwin/MSYS)"
    
    # ----------------------------------------------------------------------------
    # 1. Source conda so 'conda' and 'conda activate' work
    source "/c/Users/$(whoami)/anaconda3/etc/profile.d/conda.sh"

    # 2. (Optional) remove old environment if present
    conda env remove -n urbanity -y || true

    # 3. Create from environment.yml
    conda env create -n urbanity -f environment.yml

    # 4. Activate
    conda activate urbanity

    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict

    conda install mamba -c conda-forge -y

    mamba install geopandas -y
    python -m pip install urbanity==0.5.8
    # 5. Uninstall pip version of networkit (if it exists)
    python -m pip uninstall -y networkit || true

    python -m pip install setuptools==68
    python -m pip install rasterio==1.4.0
    mamba install -c conda-forge -y ipyleaflet 
    mamba install -c conda-forge -y jupyter 
    # mamba install -c conda-forge -y pyarrow
    mamba install -c conda-forge -y networkit
    mamba install -c conda-forge -y h5py

else
    echo "Unknown OS type: $OSTYPE"
    exit 1
fi

echo "All done! Environment set up successfully."
