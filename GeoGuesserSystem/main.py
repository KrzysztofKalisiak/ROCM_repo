import geopandas as gpd
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from shapely.geometry import Point

from shapely.ops import unary_union

import torch.nn as nn
import torch

from torch.utils.data import DataLoader

from shapely.plotting import plot_polygon

import cv2

import warnings
warnings.filterwarnings("ignore")

from .utils import *
from .cam import ScoreCAM
from .visualize import visualize

from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np
import datetime
import time

import tqdm

from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as v2
import gc

from pytorch_grad_cam import GradCAM, ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

def live_plot(y, time_till_finish, labels, epochs, train_acc, test_acc, figsize=(7,5), save=False):

    loss = {k:np.vstack([d[k] for d in y]) for k in y[0]}
    y = np.concatenate(list(loss.values()), 1)

    y = y / y[0, :]

    y2 = np.vstack((train_acc, test_acc)).T

    epoch = y.shape[0]

    clear_output(wait=True)

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    ax1.plot(y, label=labels)
    ax1.set_title(str(epoch)+'/'+str(epochs)+'; '+'Time till finish: '+time_till_finish)
    ax1.set_xlabel('epoch')
    ax1.legend(loc='upper left') # the plot evolves to the right
    
    ax2.plot(y2, label=['train_acc', 'val_acc'], marker='o', linestyle='dashed')
    ax2.legend(loc='lower left')

    ax1.set_ylabel('Loss fraction', color='g')
    ax2.set_ylabel('Total country level accuracy', color='b')

    ax2.grid()

    if save:
        plt.savefig(save+'.png')

    plt.show();

def plot_loss_update(epoch, epochs, mb, loss):

    loss = {k:np.vstack([d[k] for d in loss]) for k in loss[0]}

    x = np.arange(epoch+1)
    y = np.concatenate(list(loss.values()), 1)

    graphs = [[x, y[:, j]] for j in range(y.shape[1])]
    x_margin = 0.2
    y_margin = 0.05
    x_bounds = [0, epochs+x_margin]
    y_bounds = [np.min(y, axis=None)-y_margin, np.max(y, axis=None)+y_margin]

    mb.update_graph(graphs, x_bounds, y_bounds)

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
        self.train_dataset = None
        self.test_dataset = None
        self.loss = None
        self.tau = None
        self.pct_n = None
        self.pct = None
        self.shp_n = None
        self.shp = None
        self.device = None
        self.optimizer = None
        self.loss_multiplier = None
        self.y_variable_names = None
        self.batch_size = None

    def prepare_system(self, countries=['AL']):
        
        self.full_precompute = torch.tensor(f(self.pct_n, self.shp_n)).to(torch.float64).to(self.device)
        self.criterions = self.loss

        self.select_indexes = np.arange(self.shp.index.shape[0])[self.shp.index.str.startswith(tuple(countries))]
        self.selected_indexes = torch.Tensor(self.select_indexes).to(torch.int32).to(self.device)

        self.tasks_location = { v[0]: (k, v[1]) for k, l in self.NN.tasks.items() for v in l }

    def prepare_dataloaders(self):

        if self.NN.barebone_model is None:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, 
                                            num_workers=4, persistent_workers=True, 
                                            multiprocessing_context='spawn', shuffle=True) 
        else:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) 
        
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
    
    def real_output_extract(self, data):

        real_output = {}
        
        generic_output = {}

        Hav_gi_xn = self.full_precompute[data[0], :][:, self.selected_indexes.long()]
        Hav_gn_xn = data[1][:, None].to(self.device)
        distance = Hav_gi_xn-Hav_gn_xn
        adj_distance = distance - torch.min(distance, dim=1, keepdim=True).values
        generic_output['geolocation'] = torch.exp(-(adj_distance)/self.tau)

        generic_output['side_tasks'] = [data[5][:, [j]] for j in range(data[5].shape[1])]

        # pack as instructed

        for level in self.NN.tasks:

            real_output[level] = [_ for _ in range(sum(self.NN.target_outputs[level]))]

            for task in self.NN.tasks[level]:
                
                name, position = task

                y = generic_output[name]
                if type(y) is not list:
                    y = [y]

                real_output[level][position] = y

        return real_output

    def loss_calculator(self, model, real):

        loss = {}
        for auxiliary_level in model:

            if auxiliary_level not in self.criterions:
                continue
                
            loss[auxiliary_level] = []

            for i, loss_func in enumerate(self.criterions[auxiliary_level]):

                loss[auxiliary_level].append(self.loss_multiplier[auxiliary_level][i]* \
                            loss_func(model[auxiliary_level][i], real[auxiliary_level][i]))
                
        return loss
    
    def accuracy_calc(self):

        self.generate_test_main(on='test')
        tt = self.task_summary['geolocation'].copy()
        tt = tt.applymap(lambda x: x[:12])
        tt = tt.groupby('real')['pred'].value_counts().unstack(-1)
        tt = tt.reindex(tt.index, axis=1).fillna(0)
        test_acc = np.diag(tt).sum()/np.sum(tt.values, axis=None)

        self.generate_test_main(on='train')
        tt = self.task_summary['geolocation'].copy()
        tt = tt.applymap(lambda x: x[:12])
        tt = tt.groupby('real')['pred'].value_counts().unstack(-1)
        tt = tt.reindex(tt.index, axis=1).fillna(0)
        train_acc = np.diag(tt).sum()/np.sum(tt.values, axis=None)

        return test_acc, train_acc

    def train(self, epochs=1, name='Unknown'):

        if self.train_dataloader is None:
            self.prepare_dataloaders()

        self.train_loss = []
        self.train_loss_granular = []

        variable_names_with_level = sum([[str(k)+'_'+y for y in self.y_variable_names[k]] for k in self.y_variable_names], [])

        start_full_training = time.time()
        test_acc, train_acc = [], []
        test_acc_, train_acc_ = 0, 0
        for epoch in tqdm.tqdm(range(epochs)):  # loop over the dataset multiple times
            
            running_loss = 0.0
            running_loss_detailed = []

            for data in self.train_dataloader:

                inputs = data[0]

                self.optimizer.zero_grad()

                real_y = self.real_output_extract(data[1])

                network_output_y = self.NN(inputs)
                granular_loss = self.loss_calculator(network_output_y, real_y)

                running_loss_detailed.append(granular_loss)

                loss = sum([sum(granular_loss[k]) for k in granular_loss])

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            self.train_loss_granular.append({k:np.array([[y.detach().cpu() for y in x[k]] for x in running_loss_detailed]).sum(0) for k in running_loss_detailed[0]})
            self.train_loss.append(running_loss)

            time_since_beginning = time.time()-start_full_training
            sec_per_epoch = time_since_beginning/(epoch+1)
            time_still_seconds = (epochs-1-epoch)*sec_per_epoch
            time_still = str(datetime.timedelta(seconds=time_still_seconds)).split('.')[0]

            if epoch % 10 == 0:
                test_acc_, train_acc_ = self.accuracy_calc()
            test_acc.append(test_acc_)
            train_acc.append(train_acc_)

            live_plot(self.train_loss_granular, time_still, variable_names_with_level, epochs, train_acc, test_acc)

        live_plot(self.train_loss_granular, time_still, variable_names_with_level, epochs, train_acc, test_acc, save=name)
    
    def generate_test_main(self, on='test'):

        self.total_occurences = []
        self.total_sidetasks = []
        self.cords_list = []

        if on=='test':
            loader = self.test_dataloader
        elif on=='train':
            loader = self.train_dataloader

        self.NN.eval()
        with torch.no_grad():
            for i, data in enumerate(loader, 0):
                inputs = data[0].to(self.device)
                true_labels_name= data[1][4]

                network_output_y = self.NN(inputs)
                model_labels_bin = torch.argmax(network_output_y[self.tasks_location['geolocation'][0]][self.tasks_location['geolocation'][1]][0], 1).cpu()
                model_labels_name = self.shp.iloc[self.selected_indexes.cpu()].iloc[model_labels_bin].index.values
                
                real_y = self.real_output_extract(data[1])
                # other tasks
                for auxiliary_level in network_output_y:

                    if auxiliary_level not in self.criterions:
                        continue

                    for i, _ in enumerate(self.criterions[auxiliary_level]):

                        if self.y_variable_names[auxiliary_level][i] == 'geolocation':
                            continue

                        self.total_sidetasks.append((data[1][0], torch.Tensor([auxiliary_level]*data[1][0].size(0)), torch.Tensor([i]*data[1][0].size(0)), network_output_y[auxiliary_level][i][:, 0], real_y[auxiliary_level][i][:, 0]))

                self.total_occurences += list(zip(true_labels_name, model_labels_name))

                cords = list(zip(data[1][2].tolist(), data[1][3].tolist()))
                self.cords_list += cords
            
        self.NN.train()
        
        if 'side_tasks' in self.tasks_location:
            x1 = np.concatenate([np.vstack([y.detach().cpu().numpy() for y in x]) for x in self.total_sidetasks], 1)
            x2 = pd.DataFrame(x1.T, columns=['id', 'aux_lvl', 'lvl', 'pred', 'real'])
            x2['lvl'] = x2['lvl'].apply(lambda x: self.y_variable_names[self.tasks_location['side_tasks'][0]][int(x)])
            x2 = x2.drop(columns='aux_lvl').set_index(['id', 'lvl']).unstack(-1)
            x2.columns = x2.columns.swaplevel(0, 1)
            self.side_task_summary = x2.sort_index(axis=1)

            geolocation_task = pd.DataFrame(self.total_occurences, 
                                            columns=pd.MultiIndex.from_tuples([('geolocation', 'real'), ('geolocation', 'pred')]), 
                                            index=self.side_task_summary.index)
            
            self.task_summary = self.side_task_summary.join(geolocation_task).sort_index(axis=1)
        else:
            self.task_summary = pd.DataFrame(self.total_occurences, 
                                            columns=pd.MultiIndex.from_tuples([('geolocation', 'real'), ('geolocation', 'pred')]))

    def extract_photo(self, idx):

        if self.train_dataloader.dataset.load_embeddings:
            print('To extract photo full model is needed, this one is on embeddings')
            return None
            
        try:
            id = np.where(self.train_dataloader.dataset.id_translator == idx)[0][0]
            dataset = self.train_dataloader.dataset
            source = 'train'
        except:
            id = np.where(self.test_dataloader.dataset.id_translator == idx)[0][0]
            dataset = self.test_dataloader.dataset
            source = 'test'

        if dataset[id][0].dim()==4:
            res = []
            for j in range(3):

                A = dataset[id][0][:, :, :, j].cpu().numpy().transpose(1, 2, 0)
                A -= A.min()
                A /= A.max()
                res.append(A)

            res = np.hstack(res)
            fig = plt.figure(figsize = (10,10))
            plt.imshow(res, interpolation='nearest')
            plt.title(source+' '+dataset[id][1][4])
        else:
            res = dataset[id][0][:, :, :].cpu().numpy().transpose(1, 2, 0)
            fig = plt.figure(figsize = (10,10))
            plt.imshow(res, interpolation='nearest')
            plt.title(source+' '+dataset[id][1][4])
        
        return fig

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

    def asses_photo(self, data):

        inputs = data[0].to(self.device)[None, ...]

        self.NN.eval()
        with torch.no_grad():
            network_output_y = self.NN(inputs)
        self.NN.train()

        Hav_gi_xn = self.full_precompute[data[1][0], :][self.selected_indexes.long()]
        Hav_gn_xn = data[1][1]
        distance = Hav_gi_xn-Hav_gn_xn

        adj_distance = distance - torch.min(distance, dim=0, keepdim=True).values
        standard_penalize = -(adj_distance-adj_distance.mean())/adj_distance.std()
        standard_penalize += (-standard_penalize).max()

        s1 = self.shp.iloc[self.selected_indexes.cpu()]
        s1['penalize'] = standard_penalize.cpu()
        s1['probability'] = network_output_y[self.tasks_location['geolocation'][0]][self.tasks_location['geolocation'][1]][0][0].detach().cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(60, 30))

        ax1 = s1.plot(ax=ax[0], column='penalize', cmap='bone', edgecolors='black', legend=True, linewidth=0.2)
        ax1.scatter(data[1][2], data[1][3], edgecolors='black', linewidth=0.2)
        ax1.legend(title='Loss', fontsize=15)

        ax2 = s1.plot(ax=ax[1], column='probability', cmap='OrRd', edgecolors='black', legend=True, linewidth=0.2)
        ax2.scatter(data[1][2], data[1][3], edgecolors='black', linewidth=0.2)
        ax2.legend(title='probability', fontsize=15)
        
        return fig

    def gradient_track(self, data, auxiliary_lvl=3, lvl=0, cam_type='ScoreCam'):

        class CNet(nn.Module):
            def __init__(self, NN):
                super().__init__()
                self.NN = NN
            def forward(self, x):
                return self.NN(x)[auxiliary_lvl][lvl]
        im = CNet(self.NN)
        im.eval()

        for name, param in im.named_parameters():
            param.requires_grad = True

        self.prepare_dataloaders()

        #data = BR.test_dataloader.dataset[2355]
        original_photo_path = self.test_dataloader.dataset.img_dir.loc[data[1][0]]['path']
        t = v2.ToTensor()

        torch.cuda.empty_cache()
        gc.collect()

        def reshape_transform(tensor, height=27, width=27):
            result = tensor.reshape(tensor.size(0),
                height, width, tensor.size(2))
            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        target_layers = [im.NN.barebone_model.trunk.blocks[-2].norm2,
                         im.NN.barebone_model.trunk.blocks[-1].norm2]
        
        types_cam = {'GradCAM':GradCAM, 
                    'ScoreCAM':ScoreCAM,
                    'GradCAMPlusPlus':GradCAMPlusPlus,
                    'AblationCAM':AblationCAM,
                    'XGradCAM':XGradCAM,
                    'EigenCAM':EigenCAM,
                    'EigenGradCAM':EigenGradCAM,
                    'LayerCAM':LayerCAM,
                    'FullGrad':FullGrad}

        cam = types_cam[cam_type](model=im.to('cuda'), target_layers=target_layers, reshape_transform=reshape_transform)

        images = {}

        for i, angle in enumerate(['120', '240', '360']):

            original_photo = Image.open(original_photo_path.replace('*', angle))
            original_photo_full = t(original_photo)

            grayscale_cam = cam(data[0][np.newaxis, :, :, :, i].to('cuda'), aug_smooth=True)

            mask = np.swapaxes(np.swapaxes(grayscale_cam, 0, 1), 1, 2)
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = np.float32(heatmap) / 255

            r = v2.Resize((640, 640))

            heatmap_full = r(torch.Tensor(heatmap).swapaxes(1, 2).swapaxes(0, 1))
            p = 0.6
            cam_ = (1 - p) * heatmap_full + p * original_photo_full
            cam_ = cam_ / torch.max(cam_)
            images[angle] = to_pil_image(torch.concatenate([cam_,original_photo_full], axis=2))

        return images