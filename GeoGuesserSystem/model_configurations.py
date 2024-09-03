import torch.nn as nn
import torchvision.transforms as v2

from .utils import *

model_configs = {

    'ID1':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
                                        nn.ReLU(),
                                        nn.Linear(1024, 1500),
                                        nn.ReLU(),
                                        nn.Linear(1500, 1001),
                                        nn.Softmax(dim=1)
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':v2.Resize((224, 224))
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