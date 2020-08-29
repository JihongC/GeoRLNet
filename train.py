import torch

import tianshou as ts
from netEnv import NetEnv, TransObservation
from model import DQNNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_envs = ts.env.VectorEnv([(lambda: TransObservation(NetEnv())) for _ in range(50)])
test_envs = ts.env.VectorEnv([lambda: TransObservation(NetEnv()) for _ in range(10)])

net = DQNNet(3, device).to(device)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)
policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(size=20000))
test_collector = ts.data.Collector(policy, test_envs)

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10000, step_per_epoch=1000, collect_per_step=10,
    episode_per_test=20, batch_size=64,
    train_fn=lambda e: policy.set_eps(0.1),
    test_fn=lambda e: policy.set_eps(0.05),
    stop_fn=None,
    writer=None)
print(f'Finished training! Use {result["duration"]}')
