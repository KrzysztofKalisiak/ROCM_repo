import torch.nn as nn
from torchvision.models.feature_extraction import get_graph_node_names


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