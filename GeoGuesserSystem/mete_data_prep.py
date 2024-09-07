import rioxarray
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
from shapely import Point

countries_t = system_configs[SYSTEM_ID]['COUNTRIES_T']

countries_all = set([x.split(GLOBAL_DATA_PATH.rsplit('/', 1)[1]+'/')[1].split('/')[0] for x in glob.glob(GLOBAL_DATA_PATH+'/*/*/*.jpg')])
if countries_t is None:
    countries_t = countries_all

PMD = PreMergerData(GLOBAL_DATA_PATH, countries_all, None)
PMD.preprocess()
DC = DataContainer(PMD.premerged_shapes, GLOBAL_DATA_PATH, countries_all)
DC.load_pictures()
DC.precalculateHaversineDist()

pct = DC.pictures
shp = PMD.premerged_shapes

pct_n = pct['geometry'].apply(lambda x: (x.x, x.y)).values
pct_n = np.array([*pct_n])

shp_n = shp['geometry'].apply(lambda x: (x.centroid.x, x.centroid.y)).values
shp_n = np.array([*shp_n])

DFO = DataFeederOperator(DC.pictures, list(countries_t), countries_all)
DFO.select_data()
DFO.train_test_split(0.2)

res = pd.DataFrame()

for var in ['rg', 'tn', 'tx', 'pr', 'ws', 'pd']:

    print(var)

    xds = rioxarray.open_rasterio('/home/krzysztof-kalisiak/Downloads/EMO-1arcmin-%s_2022.nc' % var)
    scale_factor = xds.scale_factor
    offset = xds.add_offset
    xds = xds.drop_vars('wgs_1984').mean('time').to_dataframe()
    xds = xds.loc[xds[var]>-9000]
    xds = (xds+offset)*scale_factor
    xds['geometry'] = [Point(x[1], x[0]) for x in xds.index.values]
    xds = xds.reset_index(drop=True)
    xds = gpd.sjoin_nearest(DFO.total_images[['geometry']], gpd.GeoDataFrame(xds, geometry=xds['geometry'], crs='EPSG:4326'))
    xds.set_index('geometry', inplace=True, append=True)

    res = pd.concat((res, xds), axis=1)

    res.index = res.index.droplevel(1)

    res.drop(columns='index_right').rename(columns={'pr':'precipitation', 'ws':'wind_speed', 'tx':'max_temp', 'tn':'min_temp', 'rg':'solar radiation', 'pd':'water vapour pressure'}).to_csv('DATA_OTHER/meteorological.csv')