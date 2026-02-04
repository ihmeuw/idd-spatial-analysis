"""
Tests for spatial_utils module
"""

import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from idd_spatial_analysis import spatial_utils


def test_points_from_xy():
    """Test creating GeoDataFrame from x, y coordinates"""
    df = pd.DataFrame({
        'name': ['Seattle', 'Portland', 'Vancouver'],
        'longitude': [-122.3321, -122.6765, -123.1207],
        'latitude': [47.6062, 45.5152, 49.2827]
    })
    
    gdf = spatial_utils.points_from_xy(df)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 3
    assert gdf.crs == "EPSG:4326"
    assert all(isinstance(geom, Point) for geom in gdf.geometry)


def test_reproject_gdf():
    """Test reprojecting GeoDataFrame"""
    df = pd.DataFrame({
        'name': ['Test Point'],
        'longitude': [-122.3321],
        'latitude': [47.6062]
    })
    
    gdf = spatial_utils.points_from_xy(df)
    gdf_projected = spatial_utils.reproject_gdf(gdf, target_crs="EPSG:3857")
    
    assert gdf_projected.crs == "EPSG:3857"
    assert len(gdf_projected) == len(gdf)


def test_calculate_area():
    """Test area calculation"""
    # Create a simple square polygon (1 degree on each side at equator)
    polygon = Polygon([
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
        (0, 0)
    ])
    
    gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")
    areas_km2 = spatial_utils.calculate_area(gdf, units="km2")
    
    assert len(areas_km2) == 1
    assert areas_km2[0] > 0


if __name__ == "__main__":
    pytest.main([__file__])
