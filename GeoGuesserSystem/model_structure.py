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
    
def model_loader(model_id):

    if GLOBAL_MODELS_PATH+model_id+'.pt' in glob.glob(GLOBAL_MODELS_PATH+'*.pt'):

        return torch.load(GLOBAL_MODELS_PATH+model_id+'.pt')

    model_conf = model_configs[model_id]

    baseline_model, _, _ = open_clip.create_model_and_transforms(model_conf['basemodel'], device=DEVICE)

    ModelBarebone = GeoBrainNetwork(baseline_model.visual, 
                                    model_conf['unfreeze_basemodel_params_conf'],
                                    model_conf['geolocation_model_extension'],
                                    model_conf['preprocess']
                                    )
    
    ModelBarebone._freeze_barebone_paremeters()

    torch.save(ModelBarebone, GLOBAL_MODELS_PATH+model_id+'.pt')

    return ModelBarebone

def system_loader(force_override=False):

    if GLOBAL_SYSTEMS_PATH+SYSTEM_ID+'.pkl' in glob.glob(GLOBAL_SYSTEMS_PATH+'*.pkl') and not force_override:

        with open(GLOBAL_SYSTEMS_PATH+SYSTEM_ID+'.pkl', 'rb') as f:

            return pickle.load(f)

    system_conf = system_configs[SYSTEM_ID]

    train_dataloader, test_dataloader, pct_n, shp_n, pct, shp, countries = process_data()

    model = model_loader(system_configs[SYSTEM_ID]['model_ID'])

    BR = BRAIN()
    BR.NN = model
    BR.train_dataloader = train_dataloader
    BR.test_dataloader = test_dataloader
    BR.loss_function = system_conf['geo_loss_func']
    BR.tau = system_conf['tau']
    BR.pct_n = pct_n
    BR.pct = pct
    BR.shp_n = shp_n
    BR.shp = shp
    BR.device = DEVICE

    BR.prepare_system(list(countries))

    with open(GLOBAL_SYSTEMS_PATH+SYSTEM_ID+'.pkl', 'wb') as f:

        pickle.dump(BR, f)

    return BR

def save_system(BR):

    torch.save(BR.NN, GLOBAL_MODELS_PATH+system_configs[SYSTEM_ID]['model_ID']+'.pt')

    with open(GLOBAL_SYSTEMS_PATH+SYSTEM_ID+'.pkl', 'wb') as f:

        pickle.dump(BR, f)