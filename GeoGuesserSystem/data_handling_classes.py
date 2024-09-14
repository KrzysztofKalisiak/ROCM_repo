import geopandas as gpd
import pandas as pd
import numpy as np
import glob
from PIL import Image

from shapely.geometry import Point

from sklearn.cluster import OPTICS, KMeans

from shapely.ops import unary_union

import torchvision.transforms as v2
from torch.utils.data import Dataset, DataLoader


from PIL import Image

import warnings

from .utils import *

warnings.filterwarnings("ignore")

from .config import *


class PreMergerData:
    def __init__(self, full_pictures_path, countries, settings):

        self.full_pictures_path = full_pictures_path
        self.countries = countries

        self.settings = settings

        shapefile = gpd.read_file(GLOBAL_DATA_OTH_PATH+'NUTS_RG_20M_2021_4326.shp')
        shapefile.loc[shapefile['NUTS_ID']=='RO21', 'NAME_LATN'] = 'Nord-Est RO' # there is another nord-est in italy
        self.shapefile = shapefile.loc[shapefile['LEVL_CODE']==3][['NUTS_ID', 'geometry']]
        self.shapefile['NUTS_ID'] = self.shapefile['NUTS_ID'].apply(lambda x: '_'.join([x[:2], x[2], x[3], x[4]]))

        self.centroid_distance_matrix = self.shapefile.to_crs('+proj=cea').centroid.to_crs('EPSG:4326').apply(lambda g: self.shapefile.distance(g));
        self.touching = self.shapefile.to_crs('+proj=cea').centroid.to_crs('EPSG:4326').apply(lambda g: self.shapefile.touches(g));
        self.country = self.shapefile['NUTS_ID'].apply(lambda x: x[:2]==self.shapefile['NUTS_ID'].str[:2])
    
    def preprocess(self):
        total_c = pd.DataFrame()

        for c in self.countries:
            y = [[x, Point(list(map(float, x.rsplit('/', 1)[1].rsplit('|', 3)[0].split('|'))))] for x in glob.glob(self.full_pictures_path+'/%s/*.jpg' % c)]

            y = gpd.GeoDataFrame(y).rename(columns={0:'path', 1:'geometry'})
            y = y.set_geometry('geometry', crs='EPSG:4326')
            y = gpd.sjoin(y, self.shapefile).drop(columns='index_right').copy()

            to_be_merged = y.groupby('NUTS_ID')['NUTS_ID'].count()

            y['add_cat2'] = np.nan
            y['add_cat2'] = y['add_cat2'].astype(str)

            merges = []

            for x in to_be_merged.loc[to_be_merged<30].index:
                ind = self.shapefile['NUTS_ID'].loc[self.shapefile['NUTS_ID']==x].index[0]
                distances = self.centroid_distance_matrix[ind].drop(ind).copy()
                touch_ar = self.touching[ind].drop(ind).copy()
                self_country_ar = self.country[ind].drop(ind).copy()

                distances += np.abs(touch_ar-1)*100000
                distances += np.abs(self_country_ar-1)*100000
                
                id_to_merge_to = self.shapefile['NUTS_ID'].loc[distances.loc[distances==distances.min()].index[0]]

                merges.append((id_to_merge_to, x))

            result = []

            for tup in merges:
                for idx, already in enumerate(result):
                    if any(item in already for item in tup):
                        result[idx] = already + tuple(item for item in tup if item not in already)
                        break
                else:
                    result.append(tup)

            for i, x in enumerate(result):

                for xx in x:

                    y.loc[y['NUTS_ID']==xx, 'add_cat2'] = x[0]+str(i)

            y['NUTS_ID_new'] = y.apply(lambda x: x['add_cat2'] if x['add_cat2'] != 'nan' else x['NUTS_ID'], axis=1)
            y = y.drop(columns = ['add_cat2'])

            total_c = pd.concat([total_c, y])

        x3 = self.shapefile.join(total_c[['NUTS_ID', 'NUTS_ID_new']].drop_duplicates().set_index('NUTS_ID'), on='NUTS_ID', rsuffix='_points')
        self.premerged_shapes = x3.groupby('NUTS_ID_new')['geometry'].apply(lambda x: unary_union(x))
        self.premerged_shapes.crs = "EPSG:4326"
        self.premerged_shapes = self.premerged_shapes.reset_index().set_index('NUTS_ID_new')

class DataContainer:
    def __init__(self, polygons, full_pictures_path, countries):

        self.polygons = polygons
        self.full_pictures_path = full_pictures_path
        self.countries = countries

        self.pictures_total = []
    
    def load_pictures(self):

        for c in sorted(self.countries):
            y = [[x, Point(list(map(float, x.rsplit('/', 1)[1].rsplit('|', 3)[0].split('|'))))] for x in glob.glob(self.full_pictures_path+'/%s/*.jpg' % c)]
            y = gpd.GeoDataFrame(y).rename(columns={0:'path', 1:'geometry'})
            y = y.set_geometry('geometry', crs='EPSG:4326')
            y = gpd.sjoin(y, gpd.GeoDataFrame(self.polygons, crs='EPSG:4326').reset_index())
            self.pictures_total.append(y.drop(columns='index_right'))

        self.pictures = pd.concat(self.pictures_total, ignore_index=True).rename(columns={'NUTS_ID_new':'NUTS_ID_fin'})

    def precalculateHaversineDist(self):
        a = self.pictures.join(pd.Series(self.polygons.to_crs('+proj=cea').centroid.to_crs('EPSG:4326'), name='centroids'), on='NUTS_ID_fin')
        self.pictures['distance_to_centroid'] = a.apply(lambda x: haversine_distance(
            x['geometry'].x, x['geometry'].y, x['centroids'].x, x['centroids'].y), axis=1)
        
class DataFeederOperator:
    def __init__(self, total_images, country_selector=[], countries_all=[]):
        self.total_images = total_images
        self.country_selector = country_selector
        self.countries_all = countries_all

        self.country_dictionary = {y:i for i, y in enumerate(self.countries_all)}
        self.country_dictionary_back = {v:k for k,v in self.country_dictionary.items()}

        self.panorama = False

    def select_data(self):
        if self.country_selector != []:
            self.selected_images = self.total_images.loc[self.total_images['path'].apply(lambda x: x.rsplit('/', 2)[1]).isin(self.country_selector)]
        else:
            self.selected_images = self.total_images

        if self.panorama:

            self.selected_images['path'] = self.selected_images['path'].apply(lambda x: '|'.join(['*' if i == 1 else x for i, x in enumerate(x.rsplit('|', 2))]))

            self.selected_images = self.selected_images.reset_index().groupby('path')[['index', 'geometry', 'NUTS_ID_fin', 'distance_to_centroid']].first().reset_index().set_index('index')
        
    def train_test_split(self, test_perc=0.2):
        self.test_paths = self.selected_images.sample(frac=test_perc).sort_index()
        self.train_paths = self.selected_images.drop(self.test_paths.index).sort_index()

convert_tensor = v2.ToTensor()
class GeoBrainDataset(Dataset):
    def __init__(self, img_dir, target_transform=None, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.id_translator = self.img_dir.index.values

        self.meteo_data = pd.read_csv(GLOBAL_DATA_OTH_PATH+'meteorological.csv', index_col=0)
        self.meteo_normalization_params = self.meteo_data.agg(['mean', 'std'])

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, id):
        
        idx = self.id_translator[id]

        img_path = self.img_dir.loc[idx]['path'].split('storage/')[1]

        if '*' in img_path:
            force3pano = True
        else:
            force3pano = False

        paths = [x for x in glob.glob('storage/'+img_path)]

        images = [convert_tensor(Image.open(path)) for path in paths]

        d_t_c = self.img_dir.loc[idx]['distance_to_centroid']

        if self.target_transform is not None:
          idx, d_t_c  = self.target_transform(idx, d_t_c)

        if self.transform is not None:
          images = [self.transform(image) for image in images]

        oth_data_meteo = torch.Tensor((self.meteo_data.loc[idx].values-
                                       self.meteo_normalization_params.values[0, :])/self.meteo_normalization_params.values[1, :]).to('cuda') # solar radiation,min_temp,max_temp,precipitation,wind_speed,water vapour pressure
        
        if len(images) > 1:
           images = torch.stack(images, dim=3)
        else:
           images = images[0]

        if force3pano:
            if images.dim() == 3:
                images = torch.stack((images,images,images), dim=3)
            elif images.dim() == 4 and images.shape[3] == 2:
                images = torch.cat((images,images[:, :, :, [0]]), dim=3)

        return images, (idx, d_t_c, self.img_dir.loc[idx]['geometry'].x, self.img_dir.loc[idx]['geometry'].y, self.img_dir.loc[idx]['NUTS_ID_fin'], oth_data_meteo)