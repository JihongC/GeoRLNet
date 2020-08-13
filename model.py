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


class DQNNet(nn.Module):
    def __init__(self, input_space, action_space, device):
        super(DQNNet, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_space, 4*128)
        self.fc2 = nn.Linear(4*128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 32)
        self.fc5 = nn.Linear(32, action_space)

    def forward(self, obs, state=None, info={}):
        # assert isinstance(obs, dict)
        node_features = [torch.Tensor(x[:, :, 0]*x[:, :, 1]) for x in obs['node_features'].values()]
        x = torch.cat(node_features, 1).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x, state

