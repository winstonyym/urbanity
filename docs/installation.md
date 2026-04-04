# Installation

The recommended way to install Urbanity is to **clone the repository** and run the provided setup script, which handles the full environment automatically.

---

## Requirements

- [Git](https://git-scm.com/)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/winstonyym/urbanity.git
cd urbanity
```

---

## Step 2 — Run the setup script

The script detects your OS and GPU, installs all dependencies into a conda environment, and optionally installs a deep learning backend.

```bash
chmod +x setup.sh
./setup.sh
```

!!! note "What the script does"
    1. Installs `mamba` in your conda base if not present (for faster dependency resolution).
    2. Creates a fresh `urbanity` conda environment from `environment.yml`.
    3. Installs GeoPandas via conda-forge, then installs `urbanity` and all dependencies.
    4. Installs `networkit` via conda-forge.
    5. Installs optional OSM tools (`pyrosm`, `osmium-tool`, `geemap`, etc.).
    6. Installs the selected deep learning backend (PyG or DGL).


---

## Step 3 — Activate the environment

```bash
conda activate urbanity
```

---

## Step 4 — Verify the installation

```python
import urbanity
print(urbanity.__version__)
```

---

## External API Keys (Optional)

Some features require free API tokens. Create a `.env` file in your project root:

```bash
touch .env
```

Add your tokens:

```ini title=".env"
MAPILLARY_API_SECRET=MLY|XXXXXXXXXXXXXXX|XXXXXXXXXXXXXXX
MAPILLARY_API_TOKEN=MLY|XXXXXXXXXXXXXXX|XXXXXXXXXXXXXXX
MAPBOX_API_TOKEN=pk.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

| Service | Purpose | Sign up |
|---------|---------|---------|
| [Mapillary](https://www.mapillary.com/developer/api-documentation) | Street view imagery | Free |
| [Mapbox](https://www.mapbox.com/developers) | Satellite imagery tiles | Free tier |
| [Google Earth Engine](https://code.earthengine.google.com/register) | Remote sensing layers | Free for researchers |

Authenticate Google Earth Engine:

```python
import ee
ee.Authenticate()
ee.Initialize(project="your-project-id")
```

---

## OS-Specific Notes

=== "macOS"

    The script works natively. Ensure `conda` is initialised in your shell:
    ```bash
    conda init zsh   # or bash
    ```

=== "Linux"

    The script installs `gcc` via `apt-get` before setting up the environment. Requires `sudo`.

=== "Windows"

    Run inside **Anaconda Prompt** or **Git Bash** with Anaconda on the PATH. The script expects the Anaconda installation at `C:\Users\<you>\anaconda3`.

---