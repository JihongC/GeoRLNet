import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from netEnv import NetEnv, TransObservation


class RLNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        custom_model_config = model_config.get("custom_model_config")
        # activation = custom_model_config.get("activation")
        # no_final_layer = custom_model_config.get("no_final_layer")
        sse_hiddens = [256, 128, 64]
        hiddens = [256, 256, 256, 32]
        self.num_slice = 3
        self.vf_share_layers = True
        sse_layers = []
        prev_size = int(np.product(obs_space.shape)/self.num_slice)
        for i, size in enumerate(sse_hiddens):
            if i is not len(sse_hiddens)-1:
                sse_layers.append(nn.Linear(prev_size, size))
                sse_layers.append(nn.Tanh())
            else:
                sse_layers.append(nn.Linear(prev_size, size))
            prev_size = size
        self.sse_encode_block = nn.Sequential(*sse_layers)
        prev_size *= self.num_slice
        hidden_layers = []
        for i, size in enumerate(hiddens):
            hidden_layers.append(nn.Linear(prev_size, size))
            hidden_layers.append(nn.LeakyReLU(0.1))
            prev_size = size
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.logits = nn.Linear(prev_size, num_outputs)
        if self.vf_share_layers:
            self.value_branch = nn.Linear(prev_size, 1)
        else:
            raise NotImplementedError
        self.features = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs'].float()
        squeezed_x = [obs[:, i, :].squeeze(dim=1) for i in range(self.num_slice)]
        squeezed_x = [self.sse_encode_block(sx) for sx in squeezed_x]
        x = torch.cat(squeezed_x, dim=1)
        self.features = self.hidden_layers(x)
        logits = self.logits(self.features)
        return logits, state

    def value_function(self):
        assert self.features is not None, "must call forward() first"
        return self.value_branch(self.features).squeeze(1)


def env_creator(envconfig):
    env = NetEnv()
    return TransObservation(env)
