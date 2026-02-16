# IDD Spatial Analysis

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-managed-blue.svg)](https://python-poetry.org/)

Geospatial analysis tools for IDD research, focusing on clinic proximity analysis and population coverage estimation.

## 🚀 Features

- **Clinic Proximity Analysis**: Calculate population within X km of each clinic
- **Coverage Analysis**: Estimate population with access to at least Y clinics within X km
- **Administrative Summaries**: Aggregate results by admin boundaries (Admin 0, 1, 2)
- **Raster Operations**: Work with 100m resolution population rasters in equal area projections

## 📦 Environment Setup

This project uses conda for environment management with Python 3.13 and all necessary geospatial libraries.

### Create and Activate Environment

```bash
# Clone the repository (if not already cloned)
git clone https://github.com/ihmeuw/idd-spatial-analysis.git
cd idd-spatial-analysis

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate idd-spatial-analysis

# Install the package in development mode
poetry install

# Create Jupyter kernel (for notebooks)
python -m ipykernel install --user --name=idd-spatial-analysis --display-name="Python 3.13 (idd-spatial-analysis)"
```

## 💡 Usage

### Basic Example

```python
from idd_spatial_analysis.clinic_proximity import (
    load_clinic_data,
    population_within_distance,
    population_coverage_multiple_clinics
)

# Load clinic GPS coordinates
clinics = load_clinic_data("path/to/clinics.csv")

# Calculate population within 5 km of each clinic
results = population_within_distance(
    clinics_gdf=clinics,
    population_raster_path="path/to/population_100m.tif",
    distance_km=5.0
)

# Calculate population with access to at least 1 clinic within 5 km
coverage = population_coverage_multiple_clinics(
    clinics_gdf=clinics,
    population_raster_path="path/to/population_100m.tif",
    distance_km=5.0,
    min_clinics=1
)

print(f"Coverage: {coverage['coverage_percentage']:.2f}%")
```

### Data Requirements

1. **Clinic Data**: GPS coordinates in CSV, shapefile, or GeoJSON format
   - Expected columns: `longitude`/`latitude` or `lon`/`lat` or `x`/`y`
   
2. **Population Raster**: High-resolution population data
   - Format: GeoTIFF
   - Projection: Equal area projection (e.g., Mollweide, Albers Equal Area)
   - Resolution: 100m (or as available)
   
3. **Administrative Boundaries**: Shapefiles for Admin 0, 1, and 2 levels
   - Format: Shapefile, GeoJSON, or other vector format
   - Must have a name field for labeling

### Example Notebook

See [notebooks/example_usage.ipynb](notebooks/example_usage.ipynb) for a complete working example.

## � Project Structure

```
idd-spatial-analysis/
├── src/
│   └── idd_spatial_analysis/    # Main package
│       ├── __init__.py
│       ├── spatial_utils.py     # Core spatial utilities
│       └── clinic_proximity.py  # Clinic proximity analysis functions
├── notebooks/                   # Example notebooks
│   └── example_usage.ipynb      # Complete usage example
├── tests/                       # Unit tests
│   ├── __init__.py
│   └── test_spatial_utils.py
├── pyproject.toml              # Poetry configuration
├── environment.yml             # Conda environment (Python 3.13)
└── README.md
```

## 🔑 Key Functions

### `load_clinic_data(filepath, crs="EPSG:4326")`
Load clinic GPS coordinates from various file formats.

### `population_within_distance(...)`
Calculate population within specified distance of each individual clinic.

**Returns**: DataFrame with clinic IDs and population counts.

### `population_coverage_multiple_clinics(...)`
Calculate population with access to at least `min_clinics` within `distance_km`.

**Returns**:
- Total population
- Population with access
- Coverage percentage
- Coverage raster (number of clinics within distance per pixel)
- Admin-level summaries (if boundaries provided)

### `export_coverage_raster(...)`
Export coverage raster to GeoTIFF file.

## 🛠️ Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
# Format with black
poetry run black src/

# Sort imports
poetry run isort src/
```

## 📝 License

MIT License - see LICENSE file for details.
