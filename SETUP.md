# Setup Guide for idd-spatial-analysis

## Quick Setup

### 1. Initialize Git Repository

```bash
cd idd-spatial-analysis
git init
git add .
git commit -m "Initial commit: idd-spatial-analysis package structure"
```

### 2. Create GitHub Repository

1. Go to https://github.com/ihmeuw
2. Click "New repository"
3. Name: `idd-spatial-analysis`
4. Description: "Geospatial analysis tools for IDD research"
5. Make it Public or Private as needed
6. Do NOT initialize with README (we already have one)

### 3. Push to GitHub

```bash
git remote add origin https://github.com/ihmeuw/idd-spatial-analysis.git
git branch -M main
git push -u origin main
```

## Development Setup

### Option A: Using Conda (Recommended for complex spatial dependencies)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate idd-spatial-analysis

# Install package in editable mode
pip install -e .
```

### Option B: Using Poetry

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Verifying Installation

```bash
# Run tests
poetry run pytest

# Or with conda
pytest

# Test imports
python -c "from idd_spatial_analysis import spatial_utils; print('Success!')"
```

## Next Steps

### 1. Update Package Metadata

Edit these files with your information:
- `pyproject.toml`: Update author name and email
- `DESCRIPTION.yaml`: Update author information
- `README.md`: Add more specific details about your use cases

### 2. Add More Functionality

Common modules you might want to add to `src/idd_spatial_analysis/`:
- `raster_utils.py` - Raster data processing
- `vector_utils.py` - Vector data operations
- `climate_utils.py` - Climate data specific functions
- `plotting.py` - Visualization utilities
- `data_io.py` - Data loading/saving helpers

### 3. Add Dependencies as Needed

```bash
# With poetry
poetry add package-name

# Or edit environment.yml for conda
```

Common packages for climate-health spatial analysis:
- `rioxarray` - Raster I/O with xarray
- `salem` - Geoscientific data manipulation
- `regionmask` - Create masks of regions
- `xesmf` - Universal regridder for geospatial data
- `cartopy` - Cartographic plotting

## Using in Other Projects

### Install from GitHub

```bash
pip install git+https://github.com/ihmeuw/idd-spatial-analysis.git
```

### Or add to requirements.txt

```
git+https://github.com/ihmeuw/idd-spatial-analysis.git@main
```

### Or add to pyproject.toml

```toml
[tool.poetry.dependencies]
idd-spatial-analysis = {git = "https://github.com/ihmeuw/idd-spatial-analysis.git", branch = "main"}
```

## Development Workflow

1. Create a new branch for features
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make changes and test
   ```bash
   pytest
   black src/
   isort src/
   ```

3. Commit and push
   ```bash
   git add .
   git commit -m "Add: new feature description"
   git push origin feature/my-new-feature
   ```

4. Create pull request on GitHub

## Troubleshooting

### GDAL/Rasterio Installation Issues

If you have trouble with spatial packages:

```bash
# Use conda for GDAL
conda install -c conda-forge gdal rasterio geopandas

# Or use system GDAL
sudo apt-get install gdal-bin libgdal-dev  # Ubuntu/Debian
brew install gdal  # macOS
```

### Import Errors

Make sure you've installed the package:
```bash
pip install -e .  # From the repository root
```
