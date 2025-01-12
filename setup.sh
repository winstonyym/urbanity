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

conda install geopandas -y
python -m pip install urbanity==0.5.2
python -m pip install pyarrow
python -m pip install rasterio==1.4.0
python -m pip install jupyter-leaflet
python -m pip install jupyter

# 5. Uninstall pip version of networkit (if it exists)
python -m pip uninstall -y networkit || true

# 6. Install conda-forge's version of networkit
conda install -c conda-forge -y networkit

echo "All done! Environment set up successfully."
