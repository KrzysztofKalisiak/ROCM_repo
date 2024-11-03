from shapely import Polygon
import geopandas as gpd


polys_bound = gpd.GeoSeries([Polygon([x for x in [(-14, 36), (38, 36), (32, 66), (-11, 66)]])])
OPERATIONAL_BOUND= gpd.GeoDataFrame({'geometry': polys_bound}, crs='EPSG:4326')

GLOBAL_DATA_PATH = '/home/krzysztof-kalisiak/Desktop/ROCM_repo/storage'
GLOBAL_DATA_OTH_PATH = '/home/krzysztof-kalisiak/Desktop/ROCM_repo/DATA_OTHER/'

GLOBAL_MODELS_PATH = '/home/krzysztof-kalisiak/Desktop/ROCM_repo/_MODELS_/'

GLOBAL_SYSTEMS_PATH = '/home/krzysztof-kalisiak/Desktop/ROCM_repo/_SYSTEMS_/'

DEVICE = 'cuda'

SYSTEM_ID = 'SYS5'

PANORAMA = True