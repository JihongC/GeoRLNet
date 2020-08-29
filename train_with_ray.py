import torch.nn as nn

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from use_ray import RLNetwork, env_creator

ModelCatalog.register_custom_model("my_model", RLNetwork)
register_env("my_env", env_creator)

ray.shutdown()
ray.init()

trainer = ppo.PPOTrainer(env="my_env", config={"framework": "torch", "num_workers": 15, "num_envs_per_worker": 2,
                                               "num_gpus": 1,
                                               "model": {
                                                   "custom_model": "my_model",
                                                   # Extra kwargs to be passed to your model's c'tor.
                                                   "custom_model_config": {},
                                               },
                                               })

while True:
    print(trainer.train())
