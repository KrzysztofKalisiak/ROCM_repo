from .data_handling_classes import *
from .config import *
from .model_configurations import *

def process_data():

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

    GBD = GeoBrainDataset(DFO.train_paths)
    GBD_t = GeoBrainDataset(DFO.test_paths)

    train_dataloader = DataLoader(GBD, batch_size=15, shuffle=True)
    test_dataloader = DataLoader(GBD_t, batch_size=15, shuffle=True)

    return train_dataloader, test_dataloader, pct_n, shp_n, pct, shp, countries_t