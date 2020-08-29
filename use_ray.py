import numpy as np
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class RLNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        raise NotImplementedError

