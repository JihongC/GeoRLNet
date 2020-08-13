from torch_geometric.datasets import Planetoid

import numpy as np
import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from topology import Topology


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-6: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 6: Return new node embeddings.
        return aggr_out


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.layer1 = GCNConv(2, 2)
        self.layer2 = GCNConv(2, 2)
        self.layer3 = GCNConv(2, 2)
        self.layer4 = GCNConv(2, 2)
        self.layer5 = GCNConv(2, 2)
        self.layer6 = GCNConv(2, 2)
        self.layer7 = GCNConv(2, 2)
        self.layer8 = GCNConv(2, 2)
        self.layer9 = GCNConv(2, 2)
        self.layer10 = GCNConv(2, 2)

    def forward(self, x, edge):
        x = self.layer1(x, edge)
        x = self.layer2(x, edge)
        x = self.layer3(x, edge)
        x = self.layer4(x, edge)
        x = self.layer5(x, edge)
        x = self.layer6(x, edge)
        x = self.layer7(x, edge)
        x = self.layer8(x, edge)
        x = self.layer9(x, edge)
        x = self.layer10(x, edge)
        return x


net = Topology()
num_nodes = len(net.graph.nodes)
num_edges = len(net.graph.edges)
nodefeatures = np.zeros([len(net.graph.nodes), 2], dtype=np.float)
edges = np.zeros([2, len(net.graph.edges)], dtype=np.long)
for node_num in range(len(net.graph.nodes)):
    nodefeatures[node_num, 0] = float(max([net.graph.edges[node_num, x]['bandwidth'] for x in list(net.graph.adj[node_num])]))
    nodefeatures[node_num, 1] = float(max([net.graph.edges[node_num, x]['bandwidth'] for x in list(net.graph.adj[node_num])]))
for i, (a, b) in enumerate(list(net.graph.edges)):
    edges[0, i] = a
    edges[1, i] = b
    # edges[0, i+len(net.graph.edges)] = float(b)
    # edges[1, i+len(net.graph.edges)] = float(a)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

nodefeatures = torch.Tensor(nodefeatures).to(device)
edges = torch.Tensor(edges).to(dtype=torch.long).to(device)

ans = model(nodefeatures, edges)
