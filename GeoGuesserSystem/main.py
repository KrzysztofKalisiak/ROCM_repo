import geopandas as gpd
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from shapely.geometry import Point

from shapely.ops import unary_union

import torch.optim as optim
import tqdm

import torch.nn as nn
import torch

from shapely.plotting import plot_polygon

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
from torchvision.models.feature_extraction import get_graph_node_names
from PIL import Image
import cv2

import warnings
warnings.filterwarnings("ignore")

from .utils import *

def get_guess_matrix(total_occurences):
    df1 = pd.DataFrame(total_occurences, columns=['true', 'pred']).groupby('true')['pred'].value_counts().unstack()
    df1 = df1.reindex(df1.index, axis=1).fillna(0)
    df1 = df1.sort_index(axis=0).sort_index(axis=1)
    df1['TOTAL'] = df1.sum(1)
    df2 = df1.T
    df2['TOTAL'] = df2.sum(1)
    df1 = df2.T.copy()
    df1.columns.name = 'model best guess'
    df1.index.name = 'true label'

    return df1

def get_colors(df1):

    def get_cmap(n, name='hsv'):
        return plt.get_cmap(name, n)
    cmap = get_cmap(len(df1.index))
    colors_map = {df1.index.values[i]:cmap(i) for i in range(len(df1.index.values))}

    return colors_map

def plot_predicted_points(shp, d13, total_occurences, df1, cords_list):

    point_colors = [d13[x[1]] for x in total_occurences]

    plot_df = shp.loc[[x for x in df1.index if x != "TOTAL"]]

    bounds = plot_df['geometry'].bounds.agg({"minx":'min', "miny":'min', "maxx":'max', "maxy":'max'})
    x_span = bounds['maxx']-bounds['minx']
    y_span = bounds['maxy']-bounds['miny']
    bounds['minx'] = bounds['minx']-(x_span*0.1)
    bounds['maxx'] = bounds['maxx']+(x_span*0.1)

    bounds['miny'] = bounds['miny']-(y_span*0.1)
    bounds['maxy'] = bounds['maxy']+(y_span*0.1)


    fig, ax = plt.subplots(figsize=(10, 10))

    for i, dff in plot_df.groupby(level=0):
        plot_polygon(dff['geometry'].values[0], facecolor=d13[i], label=i, add_points=False, ax=ax)

    plt.legend(prop={'size': 8})

    points = [Point(x[0], x[1]) for x in cords_list]
    xs = [point.x for point in points]
    ys = [point.y for point in points]

    ax.scatter(xs, ys, color=point_colors, edgecolors='black')

    plt.ylim(bounds['miny'], bounds['maxy'])
    plt.xlim(bounds['minx'], bounds['maxx'])

    return fig, ax

def precision_level_set(total_occurences, shp, i=1):

    total_occurences1 = [('_'.join(x[0].split('_', i)[:i]), '_'.join(x[1].split('_', i)[:i])) for x in total_occurences]

    df1 = get_guess_matrix(total_occurences1)

    colors = get_colors(df1)

    shp1 = gpd.GeoDataFrame(shp.groupby(['_'.join(x.split("_", i)[:i]) for x in shp.index]).apply(lambda x: unary_union(x)), columns=['geometry'])

    return total_occurences1, colors, shp1, df1

class BRAIN:
    def __init__(self):
        self.NN = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.loss_function = None
        self.tau = None
        self.pct_n = None
        self.pct = None
        self.shp_n = None
        self.shp = None
        self.device = None

    def prepare_system(self, countries=['AL']):
        
        self.full_precompute = torch.tensor(f(self.pct_n, self.shp_n)).to(torch.float64)
        self.criterion = self.loss_function
        self.optimizer = optim.Adam(self.NN.parameters(), lr=0.001)

        self.select_indexes = np.arange(self.shp.index.shape[0])[self.shp.index.str.startswith(tuple(countries))]
        self.selected_indexes = torch.Tensor(self.select_indexes).to(torch.int32)

        self.NN.to(self.device)

    def train(self, epochs=1):

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in tqdm.tqdm(enumerate(self.train_dataloader, 0), total=self.train_dataloader.__len__()):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data[0].to(self.device)

                true_id_haversine = self.full_precompute[data[1][0], :][:, self.selected_indexes]
                y = torch.exp(-(true_id_haversine - (data[1][1][:, None]))/self.tau).to(self.device)

                self.optimizer.zero_grad()

                outputs = self.NN(inputs)
                loss = self.criterion(outputs, y)
                if torch.isnan(loss):
                    break
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
            print(running_loss)
    
    def test_main(self, if_plot_predicted_point):
        self.total_occurences = []
        self.cords_list = []

        for i, data in tqdm.tqdm(enumerate(self.test_dataloader, 0), total=self.test_dataloader.__len__()):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(self.device)
            true_labels_name= data[1][4]

            model_labels = self.NN(inputs)
            model_labels_bin = torch.argmax(model_labels, 1).cpu()
            model_labels_name = self.shp.iloc[self.selected_indexes].iloc[model_labels_bin].index.values

            self.total_occurences += list(zip(true_labels_name, model_labels_name))

            cords = list(zip(data[1][2].tolist(), data[1][3].tolist()))
            self.cords_list += cords

        if if_plot_predicted_point:
            
            occurences2, colors2, shp2, df12 = precision_level_set(self.total_occurences, self.shp, 4)
            plot_predicted_points(shp2, colors2, occurences2, df12, self.cords_list);

    def asses_photos(self, data):

        results = []
        labels = []

        for ind in range(data[0].shape[0]):
            inputs = data[0].to(self.device)

            true_id_haversine = self.full_precompute[data[1][0], :][:, self.selected_indexes]
            y = torch.exp(-(true_id_haversine - (data[1][1][:, None]))/self.tau).to(self.device)

            model_labels = self.NN(inputs)
            model_labels_bin = torch.argmax(model_labels, 1).cpu()
            labels.append(model_labels_bin[ind])

            standard_penalize = -(y[ind]-y[ind].mean())/y[ind].std()
            standard_penalize += (-standard_penalize).max()

            s1 = self.shp.iloc[self.selected_indexes]
            s1['penalize'] = standard_penalize.cpu()
            s1['probability'] = model_labels[0].detach().cpu().numpy()

            fig, ax = plt.subplots(1, 2)

            cords = list(zip(data[1][2].tolist(), data[1][3].tolist()))

            ax1 = s1.plot(ax=ax[0], column='penalize', cmap='bone', edgecolors='black', legend=True, linewidth=0.2)
            ax1.scatter(cords[ind][0], cords[ind][1], edgecolors='black', linewidth=0.2)
            ax1.legend(title='Loss', fontsize=15)

            ax2 = s1.plot(ax=ax[1], column='probability', cmap='OrRd', edgecolors='black', legend=True, linewidth=0.2)
            ax2.scatter(cords[ind][0], cords[ind][1], edgecolors='black', linewidth=0.2)
            ax2.legend(title='probability', fontsize=15)

            results.append(fig)

        results_grad = []

        for param in self.NN.parameters():
            param.requires_grad = True

        for ind in range(data[0].shape[0]):
            inputs = data[0].to(self.device)

            self.NN.eval()
            targets = [ClassifierOutputTarget(labels[ind])] 
            target_layers = [self.NN.barebone_model.conv1]

            cam = GradCAM(model=self.NN, target_layers=target_layers) # use GradCamPlusPlus class

            # Preprocess input image, get the input image tensor
            img = np.swapaxes(np.swapaxes(inputs.detach().cpu().numpy()[ind, :, :, :], 0, 2), 0, 1)
            input_tensor = preprocess_image(img)

            # generate CAM
            grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
            cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)

            cam = np.uint8(255*grayscale_cams[0, :])
            cam = cv2.merge([cam, cam, cam])

            # display the original image & the associated CAM
            images = np.hstack((np.uint8(255*img), cam_image))
            results_grad.append(Image.fromarray(images))

        self.NN._freeze_barebone_paremeters()

        return results, results_grad
    
class EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class GeoBrainNetwork(nn.Module):
    def __init__(self, 
                 barebone_model=EmptyModel(),
                 barebone_unfreeze_config=slice(0,0),
                 model_elements=[],
                 preprocess_func=lambda x: x):
        super().__init__()

        self.barebone_model = barebone_model
        self.barebone_unfreeze_config = barebone_unfreeze_config

        self.mods = nn.ModuleList(model_elements)

        self.preprocess_func = preprocess_func

    def _freeze_barebone_paremeters(self):

        not_to_freeze = get_graph_node_names(self.barebone_model)[self.barebone_unfreeze_config]

        for name, param in self.barebone_model.named_parameters():
            if not name in not_to_freeze:
                param.requires_grad = False
        
    def forward(self, x):

        x = self.preprocess_func(x)
        x = self.barebone_model(x)

        if type(x) is tuple:
            x = x[0]

        for m in self.mods:
            x = m(x)

        return x