import torch.nn as nn
import torchvision.transforms as v2

from .utils import *

import torch.optim as optim

cntr_weight = torch.Tensor(np.array([0.02307946, 0.00701481, 0.00900437, 0.00605939, 0.01339066,
       0.007641  , 0.00197439, 0.01220628, 0.01352456, 0.0063735 ,
       0.00146655, 0.00314525, 0.00117441, 0.0118845 , 0.00648349,
       0.00955124, 0.00208584, 0.00936604, 0.14088085, 0.0095783 ,
       0.0536689 , 0.07513645, 0.45081872, 0.01263978, 0.0031718 ,
       0.00180713, 0.00778168, 0.00461274, 0.01198986, 0.0017442 ,
       0.05635234, 0.01921103, 0.00314671, 0.00203377])).to('cuda')


def first(l, _):
    return l[0]


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
        'preprocess':None,
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
                                            [nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(20007, 10000)]
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
                                            [nn.Linear(1152, 6000), nn.ReLU()]
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
                                            [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(9007, 1188)]
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
    'SYS5_small':{
        "auxiliary_loss":{
                          0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                          1:[nn.CrossEntropyLoss(weight=cntr_weight)]
                        },
        "loss_multiplier":{
                          0:[1, 1, 1, 1, 1, 1, 1],
                          1:[1]
                        },
        "tau":150,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID5_small',
        'predefined_region_grid':'ID2',
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
        'model_ID':'ID5_full',
        'predefined_region_grid':None,
        'on_embeddings':False,
        'variable_names':{
            1:['solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'],
            2:['geolocation'] # 'solar radiation','min_temp','max_temp','precipitation','wind_speed','water vapour pressure', 'GDP'
        },
        'batch_size':5
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