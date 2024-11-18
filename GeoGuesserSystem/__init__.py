
GLOBAL_DATA_PATH = '/home/krzysztof-kalisiak/Desktop/ROCM_repo/DATA'
GLOBAL_DATA_OTH_PATH = '/home/krzysztof-kalisiak/Desktop/ROCM_repo/DATA_OTHER/'

GLOBAL_MODELS_PATH = '/home/krzysztof-kalisiak/Desktop/ROCM_repo/_MODELS_/'

GLOBAL_SYSTEMS_PATH = '/home/krzysztof-kalisiak/Desktop/ROCM_repo/_SYSTEMS_/'

DEVICE = 'cuda'

#SYSTEM_ID = 'SYS1'


from .config import *

from .data_handling_classes import *

from .DataProcess import *

from .main import *

from .model_configurations import *

from .model_structure import *

from .utils import *

#from .cam import *

#from .visualize import *
