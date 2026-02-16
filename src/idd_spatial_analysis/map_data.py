@dataclass
class MapData:
    crs: str
    distance_raster: dict      # {'data': ndarray, 'meta': dict}
    admin0_gdf: GeoDataFrame
    admin1_gdf: GeoDataFrame
    admin2_gdf: GeoDataFrame
    clinics_gdf: GeoDataFrame
    neighbor_gdfs: dict        # {'IND': gdf, 'AFG': gdf, ...}
    water_gdf: GeoDataFrame    # or None