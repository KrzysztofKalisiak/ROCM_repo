import torch.nn as nn

from .utils import *

import torch.optim as optim

def first(l, _):
    return l[0]

model_configs = {
    'ID_grid0':{
        'basemodel':None,
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.1), nn.Linear(1152, 2000)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(2000+7, 10000)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(10000, 1188)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 6))],
                            2:[('geolocation', slice(0, 1))]
                            }
    },
    'ID_grid0_full':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.1), nn.Linear(1152, 2000)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(2000+7, 10000)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(10000, 1188)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 6))],
                            2:[('geolocation', slice(0, 1))]
                            }
    }
    }

system_configs = {

    'SYS_ID_grid0':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[10, 10, 10, 10, 10, 10, 10],
                        2:[1]
                        },
        "tau":200,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid0',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384_mix',
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation']
        },
        'batch_size':1024
    },
    'SYS_ID_grid0_clean':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[10, 10, 10, 10, 10, 10, 10],
                        2:[1]
                        },
        "tau":200,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid0',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation']
        },
        'batch_size':1024
        },
    'SYS_ID_grid0_full':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[10, 10, 10, 10, 10, 10, 10],
                        2:[1]
                        },
        "tau":200,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid0',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation']
        },
        'batch_size':1024
        }
}



















































































'''
def scenario_generator():

    confs = {}
    systems = {}

    i = -1

    for dr in [0.1]:

        for neurons3 in [2000]:

            for neurons4 in [10000]:

                for aux_weight in [10]:
                    
                    i += 1
                    id = 'ID_grid%i' % i

                    conf = {
                            'basemodel':None,
                            'geolocation_model_extension':[
                                [
                                    [nn.Dropout(p=dr), nn.Linear(1152, neurons3)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)]
                                ],
                                [ 
                                    [nn.ReLU(), nn.Dropout(p=dr), nn.Linear(neurons3+7, neurons4)]
                                ],
                                [ 
                                    [nn.ReLU(), nn.Dropout(p=dr), nn.Linear(neurons4, 1188)]
                                ],
                                [
                                    [nn.Softmax(dim=1)]
                                ]
                            ],
                        'unfreeze_basemodel_params_conf':slice(0, 0),
                        'preprocess':None,
                        'optimizer':optim.AdamW,
                        'optimizer_params':{'lr':0.0001},
                        'target_outputs':{
                                            0:[False, True, True, True, True, True, True, True],
                                            1:[False],
                                            2:[True],
                                            3:[True]
                                        },
                        'concurrent_reduction':{
                                                0:torch.cat,
                                                1:first,
                                                2:first,
                                                3:first
                                                },
                        'tasks':{
                                                0:[('side_tasks', slice(0, 6))],
                                                2:[('geolocation', slice(0, 1))]
                                                }
                        }

                    confs[id] = conf

                    system_i = {
                        "auxiliary_loss":{
                                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                                        2:[nn.CrossEntropyLoss()]
                                        },
                        "loss_multiplier":{
                                        0:[aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight],
                                        2:[1]
                                        },
                        "tau":200,
                        'COUNTRIES_T':None,
                        'blur_system':None,
                        'save_system':True,
                        'model_ID':id,
                        'predefined_region_grid':None,
                        'on_embeddings':'ViT-SO400M-14-SigLIP-384_mix',
                        'variable_names':{
                            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
                            2:['geolocation']
                        },
                        'batch_size':1024
                        }
                    
                    systems['SYS_'+id] = system_i

    return systems, confs


#system_configs, model_configs = scenario_generator()
'''
'''
def scenario_generator():

    confs = {}
    systems = {}

    i = 0

    for dr in [0.2, 0.4, 0.5]:

        for neurons3 in [2000, 4000, 10000]:

            for neurons4 in [2000, 4000, 10000]:

                for aux_weight in [2]:
                    
                    i += 1
                    id = 'ID_grid%i' % i
                    if i < 30:
                        continue

                    conf = {
                            'basemodel':None,
                            'geolocation_model_extension':[
                                [
                                    [nn.Dropout(p=dr), nn.Linear(1152, neurons3)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)],
                                    [nn.Dropout(p=dr), nn.Linear(1152, 1)]
                                ],
                                [ 
                                    [nn.ReLU(), nn.Dropout(p=dr), nn.Linear(neurons3+7, neurons4)]
                                ],
                                [ 
                                    [nn.ReLU(), nn.Dropout(p=dr), nn.Linear(neurons4, 1188)]
                                ],
                                [
                                    [nn.Softmax(dim=1)]
                                ]
                            ],
                        'unfreeze_basemodel_params_conf':slice(0, 0),
                        'preprocess':None,
                        'optimizer':optim.AdamW,
                        'optimizer_params':{'lr':0.0001},
                        'target_outputs':{
                                            0:[False, True, True, True, True, True, True, True],
                                            1:[False],
                                            2:[True],
                                            3:[True]
                                        },
                        'concurrent_reduction':{
                                                0:torch.cat,
                                                1:first,
                                                2:first,
                                                3:first
                                                },
                        'tasks':{
                                                0:[('side_tasks', slice(0, 6))],
                                                2:[('geolocation', slice(0, 1))]
                                                }
                        }

                    confs[id] = conf

                    system_i = {
                        "auxiliary_loss":{
                                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                                        2:[nn.CrossEntropyLoss()]
                                        },
                        "loss_multiplier":{
                                        0:[aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight, aux_weight],
                                        2:[1]
                                        },
                        "tau":200,
                        'COUNTRIES_T':None,
                        'blur_system':None,
                        'save_system':True,
                        'model_ID':id,
                        'predefined_region_grid':None,
                        'on_embeddings':'ViT-SO400M-14-SigLIP-384_mix',
                        'variable_names':{
                            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
                            2:['geolocation']
                        },
                        'batch_size':1024
                        }
                    
                    systems['SYS_'+id] = system_i

    return systems, confs
'''

'''
baseline_model, preprocess = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384', device='cuda')
baseline_model = baseline_model.visual

from pathlib import Path
for x_ in tqdm.tqdm(glob.glob('storage/*/*.jpg')):


    Path(x_.replace('storage', 'embeddings/ViT-SO400M-14-SigLIP-384_mix').replace('.jpg', '.pt').rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)

    x = preprocess(augmenter(Image.open(x_))).to('cuda')

    x = baseline_model(x[None, :, :, :])

    torch.save(x, x_.replace('storage', 'embeddings/ViT-SO400M-14-SigLIP-384_mix').replace('.jpg', '.pt'))

res = {}
for x in tqdm.tqdm(glob.glob('embeddings/ViT-SO400M-14-SigLIP-384_mix/*/*.pt')):

    res[x.split('384_mix/')[1]] = torch.load(x)

torch.save(res, 'embeddings/ViT-SO400M-14-SigLIP-384_mix/all_embeddings.pt')

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
                                            [nn.Linear(1500, 1), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()]
                                        ],
                                        [ 
                                            [nn.Dropout(p=0.1), nn.Linear(1008, 1188),nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
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
                                },
        'tasks':{
                                1:['side_tasks'],
                                2:['geolocation']
                                }
        },
    'ID2':{
        'basemodel':None,
        'geolocation_model_extension':[
                                        [
                                            [nn.Dropout(p=0.1),nn.ReLU(),nn.Linear(1152, 1500),nn.Dropout(p=0.1),nn.ReLU()]
                                        ], 
                                        [
                                            [nn.Linear(1500, 1200), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()],
                                            [nn.Linear(1500, 1), nn.ReLU()]
                                        ],
                                        [ 
                                            [nn.Dropout(p=0.1), nn.Linear(1207, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.00005},
        'target_outputs':{
                            1:[False, True, True, True, True, True, True, True],
                            2:[True],
                            3:[True]
                        },
        'concurrent_reduction':{
                                0:first,
                                1:torch.cat,
                                2:first,
                                3:first
                                },
        'tasks':{
                                1:['side_tasks'],
                                2:['geolocation']
                                }
        },

    'ID3':{
        'basemodel':None,
        'geolocation_model_extension':[
                                        [
                                            [nn.Linear(1152, 3000), nn.ReLU()],
                                            [nn.Linear(1152, 1), nn.ReLU()],
                                            [nn.Linear(1152, 1), nn.ReLU()],
                                            [nn.Linear(1152, 1), nn.ReLU()],
                                            [nn.Linear(1152, 1), nn.ReLU()],
                                            [nn.Linear(1152, 1), nn.ReLU()],
                                            [nn.Linear(1152, 1), nn.ReLU()],
                                            [nn.Linear(1152, 1), nn.ReLU()]
                                        ],
                                        [ 
                                            [nn.Dropout(p=0.1), nn.Linear(3007, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False, True, True, True, True, True, True, True],
                            1:[True],
                            2:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:first,
                                2:first
                                },
        'tasks':{
                                0:[('side_tasks', slice(0, 6))],
                                1:[('geolocation', slice(0, 1))]
                                }
        },
    'ID4':{
        'basemodel':None,
        'geolocation_model_extension':[
                                        [
                                            [nn.Linear(1152, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':None,
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0005},
        'target_outputs':{
                            0:[True],
                            1:[True]
                        },
        'concurrent_reduction':{
                                0:first,
                                1:first
                                },
        'tasks':{
                                0:[('geolocation', slice(0, 1))]
                                }
        },
        'ID5':{
        'basemodel':None,
        'geolocation_model_extension':[
                                        [
                                            [nn.Linear(1152, 9000)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(9007, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':None,
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False, True, True, True, True, True, True, True],
                            1:[True],
                            2:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:first,
                                2:first
                                },
        'tasks':{
                                0:[('side_tasks', slice(0, 6))],
                                1:[('geolocation', slice(0, 1))]
                                }
        },
        'ID5_small':{
        'basemodel':None,
        'geolocation_model_extension':[
                                        [
                                            [nn.Linear(1152, 9000)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(9007, 34)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':None,
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False, True, True, True, True, True, True, True],
                            1:[True],
                            2:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:first,
                                2:first
                                },
        'tasks':{
                                0:[('side_tasks', slice(0, 6))],
                                1:[('geolocation', slice(0, 1))]
                                }
        },
        'ID5_b':{
        'basemodel':None,
        'geolocation_model_extension':[
                                        [
                                            [nn.Linear(1152, 12000)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.25), nn.Linear(12007, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':None,
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False, True, True, True, True, True, True, True],
                            1:[True],
                            2:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:first,
                                2:first
                                },
        'tasks':{
                                0:[('side_tasks', slice(0, 6))],
                                1:[('geolocation', slice(0, 1))]
                                }
        },
        'ID5_b_full':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
                                        [
                                            [nn.Linear(1152, 12000)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.25), nn.Linear(12007, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':None,
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False, True, True, True, True, True, True, True],
                            1:[True],
                            2:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:first,
                                2:first
                                },
        'tasks':{
                                0:[('side_tasks', slice(0, 6))],
                                1:[('geolocation', slice(0, 1))]
                                }
        },
        'ID5_c':{
        'basemodel':None,
        'geolocation_model_extension':[
                                        [
                                            [nn.Linear(1152, 20000)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(20007, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':None,
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False, True, True, True, True, True, True, True],
                            1:[True],
                            2:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:first,
                                2:first
                                },
        'tasks':{
                                0:[('side_tasks', slice(0, 6))],
                                1:[('geolocation', slice(0, 1))]
                                }
        },
        'ID5_d':{
        'basemodel':None,
        'geolocation_model_extension':[
                                        [
                                            [nn.Dropout(p=0.4), nn.Linear(1152, 20000)],
                                            [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                                            [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                                            [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                                            [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                                            [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                                            [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                                            [nn.Dropout(p=0.4), nn.Linear(1152, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(20007, 10000)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(10000, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':None,
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False, True, True, True, True, True, True, True],
                            1:[False],
                            2:[True],
                            3:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:first,
                                2:first,
                                3:first
                                },
        'tasks':{
                                0:[('side_tasks', slice(0, 6))],
                                2:[('geolocation', slice(0, 1))]
                                }
        },
        'ID5_e':{
        'basemodel':None,
        'geolocation_model_extension':[
                                        [
                                            [nn.Linear(1152, 20000)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(20007, 10000)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.7), nn.Linear(10000, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':None,
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False, True, True, True, True, True, True, True],
                            1:[False],
                            2:[True],
                            3:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:first,
                                2:first,
                                3:first
                                },
        'tasks':{
                                0:[('side_tasks', slice(0, 6))],
                                2:[('geolocation', slice(0, 1))]
                                }
        },
        'ID5_full':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
                                        [
                                            [nn.Linear(1152, 9000)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)],
                                            [nn.Linear(1152, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(9007, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':v2.Resize((384, 384)),
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False, True, True, True, True, True, True, True],
                            1:[True],
                            2:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:first,
                                2:first
                                },
        'tasks':{
                                0:[('side_tasks', slice(0, 6))],
                                1:[('geolocation', slice(0, 1))]
                                }
        },
        'ID6':{
        'basemodel':None,
        'geolocation_model_extension':[
                                       [
                                            [nn.Dropout(p=0.1), nn.Linear(1152, 3000), nn.Dropout(p=0.2), nn.ReLU()] # , nn.Dropout(p=0.1)
                                        ],
                                        [
                                            [nn.Linear(3000, 5000)],
                                            [nn.Linear(3000, 1)],
                                            [nn.Linear(3000, 1)],
                                            [nn.Linear(3000, 1)],
                                            [nn.Linear(3000, 1)],
                                            [nn.Linear(3000, 1)],
                                            [nn.Linear(3000, 1)],
                                            [nn.Linear(3000, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(5007, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':None,
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False],
                            1:[False, True, True, True, True, True, True, True],
                            2:[True],
                            3:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:torch.cat,
                                2:first,
                                3:first
                                },
        'tasks':{
                                1:[('side_tasks', slice(0, 6))],
                                2:[('geolocation', slice(0, 1))]
                                }
        },
        'ID6_full':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
                                       [
                                            [nn.Linear(1152, 3000), nn.ReLU()]
                                        ],
                                        [
                                            [nn.Linear(3000, 5000)],
                                            [nn.Linear(3000, 1)],
                                            [nn.Linear(3000, 1)],
                                            [nn.Linear(3000, 1)],
                                            [nn.Linear(3000, 1)],
                                            [nn.Linear(3000, 1)],
                                            [nn.Linear(3000, 1)],
                                            [nn.Linear(3000, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(5007, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':v2.Resize((384, 384)),
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False],
                            1:[False, True, True, True, True, True, True, True],
                            2:[True],
                            3:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:torch.cat,
                                2:first,
                                3:first
                                },
        'tasks':{
                                1:[('side_tasks', slice(0, 6))],
                                2:[('geolocation', slice(0, 1))]
                                }
        },
        'ID7':{
        'basemodel':None,
        'geolocation_model_extension':[
                                       [
                                            [nn.Dropout(p=0.3), nn.Linear(1152, 6000), nn.Dropout(p=0.5), nn.ReLU()] # , nn.Dropout(p=0.1)
                                        ],
                                        [
                                            [nn.Linear(6000, 9000)],
                                            [nn.Linear(6000, 1)],
                                            [nn.Linear(6000, 1)],
                                            [nn.Linear(6000, 1)],
                                            [nn.Linear(6000, 1)],
                                            [nn.Linear(6000, 1)],
                                            [nn.Linear(6000, 1)],
                                            [nn.Linear(6000, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(9007, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':None,
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False],
                            1:[False, True, True, True, True, True, True, True],
                            2:[True],
                            3:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:torch.cat,
                                2:first,
                                3:first
                                },
        'tasks':{
                                1:[('side_tasks', slice(0, 6))],
                                2:[('geolocation', slice(0, 1))]
                                }
        },
        'ID7_full':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
                                       [
                                            [nn.Dropout(p=0.3), nn.Linear(1152, 6000), nn.Dropout(p=0.5), nn.ReLU()] # , nn.Dropout(p=0.1)
                                        ],
                                        [
                                            [nn.Linear(6000, 9000)],
                                            [nn.Linear(6000, 1)],
                                            [nn.Linear(6000, 1)],
                                            [nn.Linear(6000, 1)],
                                            [nn.Linear(6000, 1)],
                                            [nn.Linear(6000, 1)],
                                            [nn.Linear(6000, 1)],
                                            [nn.Linear(6000, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(9007, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':v2.Resize((384, 384)),
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False],
                            1:[False, True, True, True, True, True, True, True],
                            2:[True],
                            3:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:torch.cat,
                                2:first,
                                3:first
                                },
        'tasks':{
                                1:[('side_tasks', slice(0, 6))],
                                2:[('geolocation', slice(0, 1))]
                                }
        },
        'ID8':{
        'basemodel':None,
        'geolocation_model_extension':[
                                       [
                                            [nn.Dropout(p=0.15), nn.Linear(1152, 9000), nn.ReLU()], # , nn.Dropout(p=0.1)
                                        ],
                                        [
                                            [nn.Dropout(p=0.5), nn.Linear(9000, 10000), nn.Dropout(p=0.5), nn.ReLU()], # , nn.Dropout(p=0.1)
                                        ],
                                        [
                                            [nn.Linear(10000, 10000)],
                                            [nn.Linear(10000, 1)],
                                            [nn.Linear(10000, 1)],
                                            [nn.Linear(10000, 1)],
                                            [nn.Linear(10000, 1)],
                                            [nn.Linear(10000, 1)],
                                            [nn.Linear(10000, 1)],
                                            [nn.Linear(10000, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(10007, 5000)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(5000, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':None,
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False],
                            1:[False],
                            2:[False, True, True, True, True, True, True, True],
                            3:[True],
                            4:[True]
                        },
        'concurrent_reduction':{
                                0:first,
                                1:first,
                                2:torch.cat,
                                3:first,
                                4:first,
                                5:first
                                },
        'tasks':{
                                2:[('side_tasks', slice(0, 6))],
                                4:[('geolocation', slice(0, 1))]
                                }
        },
        'ID8_full':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
                                       [
                                            [nn.Dropout(p=0.1), nn.Linear(1152, 8000), nn.Dropout(p=0.2), nn.ReLU()] # , nn.Dropout(p=0.1)
                                        ],
                                        [
                                            [nn.Linear(8000, 10000)],
                                            [nn.Linear(8000, 1)],
                                            [nn.Linear(8000, 1)],
                                            [nn.Linear(8000, 1)],
                                            [nn.Linear(8000, 1)],
                                            [nn.Linear(8000, 1)],
                                            [nn.Linear(8000, 1)],
                                            [nn.Linear(8000, 1)]
                                        ],
                                        [ 
                                            [nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(10007, 1188)]
                                        ],
                                        [
                                            [nn.Softmax(dim=1)]
                                        ]
                                    ],
        'unfreeze_basemodel_params_conf':slice(0, 0),
        'preprocess':v2.Resize((384, 384)),
        'optimizer':optim.AdamW,
        'optimizer_params':{'lr':0.0001},
        'target_outputs':{
                            0:[False],
                            1:[False, True, True, True, True, True, True, True],
                            2:[True],
                            3:[True]
                        },
        'concurrent_reduction':{
                                0:torch.cat,
                                1:torch.cat,
                                2:first,
                                3:first
                                },
        'tasks':{
                                1:[('side_tasks', slice(0, 6))],
                                2:[('geolocation', slice(0, 1))]
                                }
        }
}

system_configs = {

    'SYS1':{
        "auxiliary_loss":{
                          1:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          2:[HaversineLoss]
                        },
        "tau":100,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID1',
        'predefined_region_grid':None,
        'on_embeddings':False
        },

    'SYS2':{
        "auxiliary_loss":{
                          1:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          2:[nn.CrossEntropyLoss()]
                        },
        "tau":100,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID2',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384'
        },

    'SYS3':{
        "auxiliary_loss":{
                          0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          1:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          0:[1, 1, 1, 1, 1, 1, 1],
                          1:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID3',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            1:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':500
        },

    'SYS4':{
        "auxiliary_loss":{
                          0:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          0:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID4',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':1024
        },

    'SYS5':{
        "auxiliary_loss":{
                          0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          1:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          0:[1, 1, 1, 1, 1, 1, 1],
                          1:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID5',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            1:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':1024
        },
    'SYS5_b':{
        "auxiliary_loss":{
                          0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          1:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          0:[2, 2, 2, 2, 2, 2, 2],
                          1:[1]
                        },
        "tau":200,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID5_b',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            1:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':1024
        },
    'SYS5_b_full':{
        "auxiliary_loss":{
                          0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          1:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          0:[2, 2, 2, 2, 2, 2, 2],
                          1:[1]
                        },
        "tau":200,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID5_b_full',
        'predefined_region_grid':None,
        'on_embeddings':None,
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            1:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':5
        },

    'SYS5_b_blur':{
        "auxiliary_loss":{
                          0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          1:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          0:[2, 2, 2, 2, 2, 2, 2],
                          1:[1]
                        },
        "tau":200,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID5_b',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384_mix',
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            1:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':1024
        },
    'SYS5_d_blur':{
        "auxiliary_loss":{
                          0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          0:[2, 2, 2, 2, 2, 2, 2],
                          2:[1]
                        },
        "tau":200,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID5_d',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384_mix',
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':1024
        },

    'SYS5_d_full':{
        "auxiliary_loss":{
                          0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          0:[2, 2, 2, 2, 2, 2, 2],
                          2:[1]
                        },
        "tau":200,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID5_d',
        'predefined_region_grid':None,
        'on_embeddings':None,
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':3
        },

    'SYS5_c':{
        "auxiliary_loss":{
                          0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          1:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          0:[2, 2, 2, 2, 2, 2, 2],
                          1:[1]
                        },
        "tau":200,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID5_c',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            1:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':1024
        },
    'SYS5_d':{
        "auxiliary_loss":{
                          0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          0:[2, 2, 2, 2, 2, 2, 2],
                          2:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID5_d',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':1024
        },
    'SYS5_e':{
        "auxiliary_loss":{
                          0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          0:[2, 2, 2, 2, 2, 2, 2],
                          2:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID5_e',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':1024
        },
    'SYS5_full':{
        "auxiliary_loss":{
                          0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          1:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          0:[1, 1, 1, 1, 1, 1, 1],
                          1:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID5_full',
        'predefined_region_grid':None,
        'on_embeddings':False,
        'variable_names':{
            0:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            1:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':1
        },
    'SYS6':{
        "auxiliary_loss":{
                          1:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          1:[1, 1, 1, 1, 1, 1, 1],
                          2:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID6',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            1:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':1024
        },
    'SYS6_full':{
        "auxiliary_loss":{
                          1:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          1:[1, 1, 1, 1, 1, 1, 1],
                          2:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID6_full',
        'predefined_region_grid':None,
        'on_embeddings':False,
        'variable_names':{
            1:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':5
        },
    'SYS7':{
        "auxiliary_loss":{
                          1:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          1:[1, 1, 1, 1, 1, 1, 1],
                          2:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID7',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            1:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':1024
        },
    'SYS7_full':{
        "auxiliary_loss":{
                          1:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          1:[1, 1, 1, 1, 1, 1, 1],
                          2:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID7_full',
        'predefined_region_grid':None,
        'on_embeddings':False,
        'variable_names':{
            1:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':5
        },
    'SYS8':{
        "auxiliary_loss":{
                          2:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          4:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          2:[2, 2, 2, 2, 2, 2, 2],
                          4:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID8',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            2:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            4:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':512
        },
    'SYS8_full':{
        "auxiliary_loss":{
                          1:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                          1:[2, 2, 2, 2, 2, 2, 2],
                          2:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID8_full',
        'predefined_region_grid':None,
        'on_embeddings':False,
        'variable_names':{
            1:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':5
    }
}
'''