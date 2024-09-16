import geopandas as gpd
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from shapely.geometry import Point

from shapely.ops import unary_union

import torch.optim as optim

import torch.nn as nn
import torch

from shapely.plotting import plot_polygon

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
from torchvision.models.feature_extraction import get_graph_node_names
from PIL import Image
import cv2

from fastprogress.fastprogress import master_bar, progress_bar

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
        self.loss = None
        self.tau = None
        self.pct_n = None
        self.pct = None
        self.shp_n = None
        self.shp = None
        self.device = None
        self.optimizer = None

    def prepare_system(self, countries=['AL']):
        
        self.full_precompute = torch.tensor(f(self.pct_n, self.shp_n)).to(torch.float64)
        self.criterions = self.loss

        self.select_indexes = np.arange(self.shp.index.shape[0])[self.shp.index.str.startswith(tuple(countries))]
        self.selected_indexes = torch.Tensor(self.select_indexes).to(torch.int32)

        self.NN.to(self.device)
    
    def real_output_extract(self, data):

        real_output = {}
        
        # level2 geolocation
        true_id_haversine = self.full_precompute[data[0], :][:, self.selected_indexes.long()]
        real_y = torch.exp(-(true_id_haversine - (data[1][:, None]))/self.tau).to(self.device)

        real_output[2] = [real_y]

        # level2 meteo+gdp data
        real_output[1] = data[5]


        return real_output

    def loss_calculator(self, model, real):

        loss = 0
        for auxiliary_level in model:

            if auxiliary_level == 1:

                for i, x in enumerate(self.criterions[auxiliary_level]):

                    loss += 10*x(model[auxiliary_level][i][:, 0], real[auxiliary_level][:, i])
            else:

                for i, x in enumerate(self.criterions[auxiliary_level]):

                    loss += x(model[auxiliary_level][i], real[auxiliary_level][i])

        return loss

    def train(self, epochs=1):

        for epoch in (pbar := master_bar(range(epochs))):  # loop over the dataset multiple times

            running_loss = 0.0
            for data in progress_bar(self.train_dataloader, parent=pbar):

                inputs = data[0].to(self.device)

                real_y = self.real_output_extract(data[1])
                network_output_y = self.NN(inputs, mode='all')
                
                self.optimizer.zero_grad()

                loss = self.loss_calculator(network_output_y, real_y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            pbar.main_bar.comment = "Loss: "+str(running_loss)
    
    def generate_test_main(self):
        self.total_occurences = []
        self.cords_list = []

        for i, data in master_bar(enumerate(self.test_dataloader, 0), total=self.test_dataloader.__len__()):
            inputs = data[0].to(self.device)
            true_labels_name= data[1][4]

            network_output_y = self.NN(inputs, mode='all')
            model_labels_bin = torch.argmax(network_output_y[2][0], 1).cpu()
            model_labels_name = self.shp.iloc[self.selected_indexes].iloc[model_labels_bin].index.values

            self.total_occurences += list(zip(true_labels_name, model_labels_name))

            cords = list(zip(data[1][2].tolist(), data[1][3].tolist()))
            self.cords_list += cords

    def metrics_and_plots(self, level=1, if_plot_predicted_point=True):
        occurences2, colors2, shp2, df12 = precision_level_set(self.total_occurences, self.shp, level)
        
        cross_table = pd.DataFrame(occurences2).groupby(0)[1].value_counts().unstack(-1).fillna(0)
        cross_table = cross_table.reindex(cross_table.index, axis=1).fillna(0)

        self.metrics = {}

        for i in cross_table.columns:
            b = cross_table.copy()
            b.columns = [i if x==i else "Z_Oth" for x in b.columns]
            b.index = [i if x==i else "Z_Oth" for x in b.index]
            
            b = b.stack().groupby(level=[0, 1]).sum().sort_index().unstack(-1)
            
            b.index.name = 'real'
            b.columns.name = 'pred'
            
            acc = (b.values[0, 0] + b.values[1, 1])/np.sum(b.values)
            
            recall = (b.values[0, 0]) / (b.values[0, 0]+b.values[0, 1])
            
            total_shots = b.values[:, 0].sum()
            total_real = b.values[0, :].sum()
            
            self.metrics[i] = [acc, recall, total_shots, total_real]

        if if_plot_predicted_point:
            
            plot_predicted_points(shp2, colors2, occurences2, df12, self.cords_list);

    def asses_photos(self, data):

        results = []
        labels = []

        inputs = data[0].to(self.device)

        true_id_haversine = self.full_precompute[data[1][0], :][:, self.selected_indexes]
        y = torch.exp(-(true_id_haversine - (data[1][1][:, None]))/self.tau).to(self.device)

        network_output_y = self.NN(inputs, mode='all')
        model_labels_bin = torch.argmax(network_output_y[2][0], 1).cpu()

        for ind in range(data[0].shape[0]): #data[0].shape[0]

            labels.append(model_labels_bin[ind])

            standard_penalize = -(y[ind]-y[ind].mean())/y[ind].std()
            standard_penalize += (-standard_penalize).max()

            s1 = self.shp.iloc[self.selected_indexes]
            s1['penalize'] = standard_penalize.cpu()
            s1['probability'] = network_output_y[2][0][ind].detach().cpu().numpy()

            fig, ax = plt.subplots(1, 2, figsize=(60, 30))

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
            target_layers = [self.NN.barebone_model.trunk.patch_embed.proj]

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