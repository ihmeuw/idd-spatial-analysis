# IDD Spatial Analysis

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-managed-blue.svg)](https://python-poetry.org/)

Repository for IDD geospatial analysis tools and utilities in Python.

## 🚀 Quick Start

### Local Installation

#### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/ihmeuw/idd-spatial-analysis.git
cd idd-spatial-analysis

# Create and activate the conda environment
conda env create -f environment.yml
conda activate idd-spatial-analysis
```

#### Option 2: Using Poetry

```bash
# Clone the repository
git clone https://github.com/ihmeuw/idd-spatial-analysis.git
cd idd-spatial-analysis

# Install dependencies with poetry
poetry install
poetry shell
```

#### Option 3: Install as a Python package (from GitHub)

You can install and use `idd_spatial_analysis` in other projects/environments:

```bash
pip install git+https://github.com/ihmeuw/idd-spatial-analysis.git
```

Then, in your Python code:

```python
from idd_spatial_analysis import spatial_utils

# Your geospatial analysis code here
```

## 📦 Package Structure

```
idd-spatial-analysis/
├── src/
│   └── idd_spatial_analysis/    # Main package
│       ├── __init__.py
│       └── spatial_utils.py     # Core spatial utilities
├── notebooks/                   # Example notebooks
├── tests/                       # Unit tests
├── pyproject.toml              # Poetry configuration
├── environment.yml             # Conda environment
└── README.md
```

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
