import torch.nn as nn
import torchvision.transforms as v2

from .utils import *

import torch.optim as optim

def first(l, _):
    return l[0].to('cuda')

model_configs = {

    'ID1':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
                                        [
                                            [nn.Dropout(p=0.1),nn.ReLU(),nn.Linear(1152, 1500),nn.Dropout(p=0.1),nn.ReLU()]
                                        ], 
                                        [
                                            [nn.Linear(1500, 1001), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()]
                                        ],
                                        [ 
                                            [nn.Dropout(p=0.1), nn.Linear(1007, 1001),nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':v2.Resize((384, 384)),
        'optimizer':optim.Adam,
        'optimizer_params':{'lr':0.001},
        'target_outputs':{
                            1:[False, True, True, True, True, True, True],
                            2:[True]
                        },
        'concurrent_reduction':{
                                0:first,
                                1:torch.cat,
                                2:first
                                }
        }
}

system_configs = {

    'SYS1':{
        "auxiliary_loss":{
                          1:[nn.MSELoss, nn.MSELoss, nn.MSELoss, nn.MSELoss, nn.MSELoss, nn.MSELoss], 
                          2:[HaversineLoss]
                        },
        "tau":100,
        'COUNTRIES_T':None,
        'train_system_epoch':5,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID1'
        }
}