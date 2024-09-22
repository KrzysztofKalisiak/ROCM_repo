import torch.nn as nn
import torch
from torchvision.models.feature_extraction import get_graph_node_names

import open_clip

from .config import *
from .model_configurations import *

from .main import BRAIN

from glob import glob

from .DataProcess import *

import pickle


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
                 preprocess_func=lambda x: x,
                 target_outputs = {},
                 reductions = {},
                 tasks = {}):
        super().__init__()

        self.barebone_model = barebone_model
        self.barebone_unfreeze_config = barebone_unfreeze_config

        self.mods = nn.ModuleList([nn.ModuleList([nn.ModuleList(y) for y in x]) for x in model_elements])

        self.preprocess_func = preprocess_func

        self.target_outputs = target_outputs

        self.reductions = reductions

        self.tasks = tasks

    def _freeze_barebone_paremeters(self):

        not_to_freeze = get_graph_node_names(self.barebone_model)[self.barebone_unfreeze_config]

        for name, param in self.barebone_model.named_parameters():
            if not name in not_to_freeze:
                param.requires_grad = False
        
    def forward(self, x):

        if x.dim() in [4, 5]: # on picture itself
        
            if x.dim() == 5: # panorama

                x_ = torch.stack([self.barebone_model(self.preprocess_func(x[:, :, :, :, ij])) for ij in range(x.shape[4])])
                x = torch.mean(x_, dim=4)

            else: # no panorama
                
                x = self.preprocess_func(x)
                x = self.barebone_model(x)
        else: # on embedding

            if x.dim() == 3: # panorama
                x = torch.mean(x, dim=2)

        outputs = {}

        for i, deep_step in enumerate(self.mods):

            if i in self.target_outputs:
                outputs[i] = []
                save = True
            else:
                save=False

            concurrent_res = []
            for ci, concurrent_step in enumerate(deep_step):
                x_ = torch.clone(x)
                for mod in concurrent_step:
                    x_ = mod(x_)
                if save:
                    if self.target_outputs[i][ci]:
                        outputs[i].append(x_)
                concurrent_res.append(x_)
            x = self.reductions[i](concurrent_res, 1)

        return outputs
    
def model_loader(model_id):

    model_conf = model_configs[model_id]

    
    if model_conf['basemodel'] is not None:
        baseline_model, _, _ = open_clip.create_model_and_transforms(model_conf['basemodel'], device=DEVICE)
        baseline_model = baseline_model.visual
    else:
        baseline_model=None

    ModelBarebone = GeoBrainNetwork(baseline_model, 
                                    model_conf['unfreeze_basemodel_params_conf'],
                                    model_conf['geolocation_model_extension'],
                                    model_conf['preprocess'],
                                    model_conf['target_outputs'],
                                    model_conf['concurrent_reduction'],
                                    model_conf['tasks']
                                    ).to(DEVICE)
    
    if model_conf['basemodel'] is not None:
        ModelBarebone._freeze_barebone_paremeters()

    Optimizer = model_conf['optimizer'](ModelBarebone.parameters(), **model_conf['optimizer_params'])

    if GLOBAL_MODELS_PATH+model_id+'.pt' in glob.glob(GLOBAL_MODELS_PATH+'*.pt'):

        checkpoint = torch.load(GLOBAL_MODELS_PATH+model_id+'.pt', weights_only=True)
        ModelBarebone.load_state_dict(checkpoint['model_state_dict'])
        Optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return ModelBarebone, Optimizer

def system_loader(force_override=False):

    if GLOBAL_SYSTEMS_PATH+SYSTEM_ID+'.pkl' in glob.glob(GLOBAL_SYSTEMS_PATH+'*.pkl') and not force_override:

        with open(GLOBAL_SYSTEMS_PATH+SYSTEM_ID+'.pkl', 'rb') as f:

            return pickle.load(f)

    system_conf = system_configs[SYSTEM_ID]

    if system_conf['predefined_region_grid'] is not None:
        premerged_shapes = gpd.read_file('predefined_region_grids/%s' % system_conf['predefined_region_grid']).set_index('NUTS_ID_ne')
    else:
        premerged_shapes = None


    train_dataloader, test_dataloader, pct_n, shp_n, pct, shp, countries = process_data(premerged_shapes, system_conf['on_embeddings'])

    model, optimizer = model_loader(system_configs[SYSTEM_ID]['model_ID'])

    BR = BRAIN()
    BR.NN = model.to(DEVICE)
    BR.train_dataloader = train_dataloader
    BR.test_dataloader = test_dataloader
    BR.loss = system_conf['auxiliary_loss']
    BR.loss_multiplier = system_conf['loss_multiplier']
    BR.tau = system_conf['tau']
    BR.pct_n = pct_n
    BR.pct = pct
    BR.shp_n = shp_n
    BR.shp = shp
    BR.device = DEVICE
    BR.optimizer = optimizer
    BR.y_variable_names = system_conf['variable_names']

    BR.prepare_system(list(countries))

    return BR

def save_system(BR):

    torch.save({
            'model_state_dict': BR.NN.state_dict(),
            'optimizer_state_dict': BR.optimizer.state_dict(),
            }, GLOBAL_MODELS_PATH+system_configs[SYSTEM_ID]['model_ID']+'.pt')

    with open(GLOBAL_SYSTEMS_PATH+SYSTEM_ID+'.pkl', 'wb') as f:

        pickle.dump(BR, f)