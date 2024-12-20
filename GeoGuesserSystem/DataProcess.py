from .data_handling_classes import *
from .config import *
from .model_configurations import *

def process_data(shp, system, on_embedding=False, preprocess_=None):

    countries_t = system_configs[system]['COUNTRIES_T']

    countries_all = set([x.split(GLOBAL_DATA_PATH.rsplit('/', 1)[1]+'/')[1].split('/')[0] for x in glob.glob(GLOBAL_DATA_PATH+'/*/*.jpg')])
    if countries_t is None:
        countries_t = countries_all

    if shp is None:

        PMD = PreMergerData(GLOBAL_DATA_PATH, countries_all, None)
        PMD.preprocess()
        shp = PMD.premerged_shapes

        shp_bounded = shp.overlay(OPERATIONAL_BOUND, how='intersection')
        shp_bounded['points'] = shp_bounded.sample_points(1)
        shp_ = shp_bounded.set_geometry('points').sjoin(shp).drop(columns='points').set_index('NUTS_ID_ne')
        shp = shp_.set_geometry('geometry')

    DC = DataContainer(shp, GLOBAL_DATA_PATH, countries_all)
    DC.load_pictures()
    DC.precalculateHaversineDist()

    pct = DC.pictures

    pct_n = pct['geometry'].apply(lambda x: (x.x, x.y)).values
    pct_n = np.array([*pct_n])

    shp_n = shp['geometry'].apply(lambda x: (x.centroid.x, x.centroid.y)).values
    shp_n = np.array([*shp_n])

    DFO = DataFeederOperator(DC.pictures, list(countries_t), countries_all)
    DFO.panorama = PANORAMA
    DFO.select_data()
    DFO.train_test_split(0.2)

    GBD = GeoBrainDataset(DFO.train_paths, load_embeddings=on_embedding, preprocess=preprocess_)
    GBD_t = GeoBrainDataset(DFO.test_paths, load_embeddings=on_embedding, preprocess=preprocess_)

    return GBD, GBD_t, pct_n, shp_n, pct, shp, countries_t
