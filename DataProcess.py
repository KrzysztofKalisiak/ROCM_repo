from data_handling_classes import *

def process_data():

    countries = set([x.split('DATA/')[1].split('/')[0] for x in glob.glob('/home/krzysztof-kalisiak/Desktop/ROCM/DATA/*/*/*.jpg')])

    PMD = PreMergerData('/home/krzysztof-kalisiak/Desktop/ROCM/DATA', countries, None)
    PMD.preprocess()

    DC = DataContainer(PMD.premerged_shapes, '/home/krzysztof-kalisiak/Desktop/ROCM/DATA', countries)
    DC.load_pictures()
    DC.precalculateHaversineDist()

    pct = DC.pictures
    shp = PMD.premerged_shapes

    pct_n = pct['geometry'].apply(lambda x: (x.x, x.y)).values
    pct_n = np.array([*pct_n])

    shp_n = shp['geometry'].apply(lambda x: (x.centroid.x, x.centroid.y)).values
    shp_n = np.array([*shp_n])

    DFO = DataFeederOperator(DC.pictures, ['AL', 'PL', 'ES', 'UK'])
    DFO.select_data()
    DFO.train_test_split(0.2)

    GBD = GeoBrainDataset(DFO.train_paths)
    GBD_t = GeoBrainDataset(DFO.test_paths)

    train_dataloader = DataLoader(GBD, batch_size=15, shuffle=True)
    test_dataloader = DataLoader(GBD_t, batch_size=15, shuffle=True)

    return train_dataloader, test_dataloader, pct_n, shp_n, pct, shp
