import torch.nn as nn
import torchvision.transforms as v2

from .utils import *

import torch.optim as optim

model_configs = {

    'ID1':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[[[
                                        nn.Dropout(p=0.1),
                                        nn.ReLU(),
                                        nn.Linear(1152, 1500)
                                    ]], [[nn.Dropout(p=0.1),
                                        nn.ReLU(),
                                        nn.Linear(1500, 1001),
                                        nn.Softmax(dim=1)]]],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':v2.Resize((384, 384)),
        'optimizer':optim.Adam,
        'optimizer_params':{'lr':0.001}
        }
}

system_configs = {

    'SYS1':{
        "geo_loss_func":HaversineLoss,
        "tau":100,
        'COUNTRIES_T':None,
        'additional_tasks':None,
        'train_system_epoch':5,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID1'
        }
}