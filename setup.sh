#!/usr/bin/env bash
set -ex  # show commands and exit on error

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
python -m pip install urbanity==0.5.3
# 5. Uninstall pip version of networkit (if it exists)
python -m pip uninstall -y networkit || true

python -m pip install setuptools==68
python -m pip install rasterio==1.4.0
mamba install -c conda-forge -y ipyleaflet 
mamba install -c conda-forge -y jupyter 
mamba install -c conda-forge -y pyarrow
mamba install -c conda-forge -y networkit

echo "All done! Environment set up successfully."
