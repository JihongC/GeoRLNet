from gym import core, spaces, ObservationWrapper
import numpy as np

from slice import Slice
from topology import Topology
from flow import Flow


class NetEnv(core.Env):
    def __init__(self, num_slices=3):
        self.num_slices = num_slices
        self.topology = Topology(1, 2, 3, 1000)
        self.slices = []
        self.flows = []
        self.num_flow_to_route = 0
        self.edge_set = None
        self.action_space = spaces.Discrete(num_slices)
        edge_feature_low = np.zeros((len(self.topology.graph.edges),), dtype=np.float)
        edge_feature_high = np.full((len(self.topology.graph.edges),), np.inf)
        self.observation_space = spaces.Dict(
            dict({'flow': spaces.Box(low=edge_feature_low, high=edge_feature_high)},
                 **{'slice' + str(i): spaces.Box(low=edge_feature_low, high=edge_feature_high)
                    for i in range(self.num_slices)}))

    def reset(self):
        """
        reset the env and return the obs

        Returns
        -------
            all slice adj_matrix and flow features
        """
        self.num_flow_to_route = 0
        # self.edge_set = self.topology.edge_set()
        self.slices = [Slice(self.topology) for _ in range(self.num_slices)]
        for s in self.slices:
            s.init_band(self.num_slices)
            s.gen_edge_features()
        flow_generator = self.topology.gen_flows()
        self.flows = [Flow(i, *next(flow_generator)) for i in range(300)]
        for f in self.flows:
            f.gen_bfs_edges(self.topology.graph)
        return dict({'flow': self.flows[self.num_flow_to_route].edge_bfs_features},
                    **{'slice' + str(i): self.slices[i].edge_features for i in range(self.num_slices)})

    def step(self, action):
        """
        Parameters
        ----------
        action: the number of slice the Agent choose to route the flow

        Returns
        -------
        obs: the updated slice matrix and features of the next flow to route
        reward : the reward of choose the slice to route compare with other slices
        done : signal to mark is all flows have been routed
        """
        # assert isinstance(action, int) and 0 <= action < self.num_slices
        self.flows[self.num_flow_to_route].belong_slice = self.slices[action]
        if len(self.flows) - 101 == self.num_flow_to_route:
            done = True
        else:
            done = False
        reward = self.calculate_reward(action)
        self.flows[self.num_flow_to_route].route_flow(self.slices[action].graph)
        self.dynamic_band_adjust(action, self.flows[self.num_flow_to_route].route_links,
                                 self.flows[self.num_flow_to_route].size, 100)
        for s in self.slices:
            s.gen_edge_features()
        if not done:
            self.num_flow_to_route += 1
        obs = dict({'flow': self.flows[self.num_flow_to_route].edge_bfs_features},
                   **{'slice' + str(i): self.slices[i].edge_features for i in range(self.num_slices)})
        return obs, reward, done, {}

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def calculate_reward(self, action):
        """
        calculate the reward for route the flow in this slice, its depend on whether the flow can be routed in other
        slices, when all slice can route the flow its reward is zero.

        Parameters
        ----------
        action : the number of slice the Agent choose to route the flow

        Returns
        -------
        reward : calculate the reward follow some rules
        """
        flow = self.flows[self.num_flow_to_route]
        result = [0 for _ in range(self.num_slices)]
        for i, s in enumerate(self.slices):
            if flow.test_flow_route(s.graph):
                result[i] = 1
        if all(result):
            return 0
        if result[action] == 0:
            zero_count = result.count(0)
            assert zero_count > 0
            return -1 * self.num_slices / zero_count
        else:
            one_count = result.count(1)
            assert one_count > 0
            return self.num_slices / one_count

    def dynamic_band_adjust(self, action, path, band_size, max_flow_size):
        """
        when route a flow successfully,implement the band adjustment between slices

        Parameters
        ----------
        action: the slice number which routed the flow
        path: the path the flow passed by
        band_size: flow size
        max_flow_size: max flow size in the top

        Returns
        -------
        void

        """
        if len(path) == 0:
            return
        else:
            for i, s in enumerate(self.slices):
                if i != action:
                    for a, b in path:
                        if s.graph[a][b]['bandwidth'] > 2 * max_flow_size:
                            self.slices[action].graph[a][b]['bandwidth'] += band_size
                            s.graph[a][b]['bandwidth'] -= band_size


class TransObservation(ObservationWrapper):
    def __init__(self, env):
        super(TransObservation, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3, 146), dtype=np.float)

    def observation(self, observation):
        bfs_edge = observation['flow']
        slice_edge = [observation['slice' + str(i)] * bfs_edge for i in range(3)]
        return np.stack(slice_edge, axis=0)
