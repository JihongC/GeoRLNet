import torch
from torch import nn
import torch.nn.functional as F

# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops, degree
#
#
# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')
#         self.lin = torch.nn.Linear(in_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#
#         x = self.lin(x)
#
#         row, col = edge_index
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#
#         return self.propagate(edge_index, x=x, norm=norm)
#
#     def message(self, x_j, norm):
#
#         return norm.view(-1, 1) * x_j
#
#
# class GCNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GCNBlock, self).__init__()
#         self.layer1 = GCNConv(in_channels, 4)
#         self.layer2 = GCNConv(4, 4)
#         self.layer3 = GCNConv(4, 8)
#         self.layer4 = GCNConv(8, out_channels)
#
#     def forward(self, x, edge_index):
#         x = self.layer1(x, edge_index)
#         x = F.relu(x)
#         x = self.layer2(x, edge_index)
#         x = F.relu(x)
#         x = self.layer3(x, edge_index)
#         x = F.relu(x)
#         x = self.layer4(x, edge_index)
#         return x


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 3, 1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu_(x)
        x = self.conv2(x)
        x = F.leaky_relu_(x)
        x = self.conv3(x)
        return x.flatten(start_dim=1)


class DQNNet(nn.Module):
    def __init__(self, action_space, device):
        super(DQNNet, self).__init__()
        self.device = device
        self.mlp = nn.Sequential(nn.Linear(146, 256), nn.LeakyReLU(0.1), nn.Linear(256, 256), nn.LeakyReLU(0.1),
                                 nn.Linear(256, 64))
        self.fc = nn.Sequential(nn.Linear(3*64, 256), nn.LeakyReLU(0.1), nn.Linear(256, 256), nn.LeakyReLU(0.1),
                                nn.Linear(256, 256), nn.LeakyReLU(0.1), nn.Linear(256, 32), nn.LeakyReLU(0.1),
                                nn.Linear(32, action_space))

    def forward(self, obs, state=None, info={}):
        # assert isinstance(obs, dict)
        x = torch.Tensor(obs).to(self.device)
        squeezed_x = [x[:, i, :].squeeze(dim=1) for i in range(3)]
        squeezed_x = [self.mlp(sx) for sx in squeezed_x]
        x = torch.cat(squeezed_x, dim=1)
        x = self.fc(x)
        return x, state

