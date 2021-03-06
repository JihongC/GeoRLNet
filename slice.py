import sys

import networkx as nx
import copy
import numpy as np


class Slice:
    def __init__(self, topology):
        self.graph = copy.deepcopy(topology.graph)
        self.node_features_min_band = None
        self.band_matrix = None
        self.edge_features = None

    def route_flow(self, flow):
        flow.route_flow(self.graph, weight='weight')

    def disconnect_flow(self, flow):
        flow.disconnect_flow(self.graph)

    def init_band(self, num_total_slice):
        """
        when init a slice, use this function to average the bandwidth

        Parameters
        ----------
        num_total_slice: the number of total slices

        Returns
        -------
        void: apply bandwidth change to this slice
        """
        for a, b in self.graph.edges:
            self.graph.edges[a, b]['bandwidth'] /= num_total_slice

    def gen_node_features(self):
        """
        generate the current node features vector of this slice, specifically, the minimum bandwidth of each node's edge
        , it will be stack with the flow features

        Returns
        -------
        void: apply the feature vector to the self.node_features_min_band, its an ndarray.
        """
        node_feature = np.zeros((len(self.graph.nodes),), dtype=np.float)
        for node in list(self.graph.nodes):
            min_band = sys.maxsize
            for i in list(self.graph[node]):
                if self.graph.edges[node, i]['bandwidth'] < min_band:
                    min_band = self.graph.edges[node, i]['bandwidth']
            node_feature[node] = min_band
        self.node_features_min_band = node_feature

    def gen_band_adj_matrix(self):
        """
        generate adjacency matrix of a specific feature, its now about bandwidth

        Returns
        -------
        void: self.band_matrix

        """
        rng = np.random.default_rng()
        band_matrix = rng.random((len(self.graph.nodes), len(self.graph.nodes)))
        for u, v in self.graph.edges:
            band_matrix[u][v] = self.graph[u][v]['bandwidth']
            band_matrix[v][u] = self.graph[u][v]['bandwidth']
        self.band_matrix = band_matrix

    def gen_edge_features(self):
        edge_feature = np.zeros((len(self.graph.edges), ), dtype=np.float)
        for i, (a, b) in enumerate(self.graph.edges):
            edge_feature[i] = self.graph[a][b]['bandwidth']
        self.edge_features = edge_feature



