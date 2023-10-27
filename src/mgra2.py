import logging
import time

import networkx as nx
import numpy as np
from sklearn.utils import Bunch
from utils import (build_co_graph, build_graph_neighbor_dict,
                   build_location_co_graph, build_location_graph,
                   build_poi_graph, build_social_graph, build_time_graph)


class MultipleGraphRandomWalk:

    def __init__(self, dataset: Bunch, algorithm_config: Bunch) -> None:
        """
        set configuration
        """
        self.dataset = dataset
        self.algorithm_config = algorithm_config
        self.graph_dicts = self._build_multiple_graphs()

    def _build_multiple_graphs(self):
        """
        build graphs with given algorithm name
        """
        graph_names = self.algorithm_config.algorithm.split("-")
        dataset = self.dataset
        friendship_old = dataset.friendship_old
        checkins = dataset.checkins
        graphs_dict = []
        graphs = []
        for graph in graph_names:
            if graph == "USG":
                graphs.append(build_social_graph(friendship_old))
            elif graph == "UCG":
                graphs.append(
                    build_co_graph(
                        checkins,
                        spatial=self.algorithm_config.spatial,
                        weight=self.algorithm_config.social_co_weight,
                        co_window=self.algorithm_config.social_co_window,
                        min_freq=self.algorithm_config.social_co_min_freq,
                    ))
            elif graph == "ULG":
                graphs.append(
                    build_location_graph(
                        checkins=checkins,
                        weight=self.algorithm_config.location_weight,
                        min_freq=self.algorithm_config.location_min_freq,
                    ))
            elif graph == "ULCG":
                merge_weight = (self.algorithm_config.location_co_weight
                                | self.algorithm_config.location_weight)
                lcg = build_location_co_graph(
                    checkins=checkins,
                    weight=self.algorithm_config.location_co_weight,
                    co_window=self.algorithm_config.location_co_window,
                    min_freq=self.algorithm_config.location_co_min_freq,
                )
                ulg = build_location_graph(
                    checkins=checkins,
                    weight=self.algorithm_config.location_weight,
                    min_freq=self.algorithm_config.location_min_freq,
                )
                lcg_edges = list(lcg.edges(data=False))
                ulg_edges = list(ulg.edges(data=False))
                ulg_edges.extend(lcg_edges)
                graph = nx.Graph()
                if merge_weight:
                    graph = nx.MultiGraph()
                graph.add_edges_from(ulg_edges)
                graphs.append(graph)
            elif graph == "UPG":
                graph = build_poi_graph(
                    checkins=checkins,
                    weight=self.algorithm_config.location_weight,
                    min_freq=self.algorithm_config.location_min_freq,
                )
                graphs.append(graph)
            elif graph == "UTG":
                graph = build_time_graph(
                    checkins=checkins,
                    weight=self.algorithm_config.location_weight,
                    min_freq=self.algorithm_config.location_min_freq,
                )
                graphs.append(graph)
        graphs_dict = [build_graph_neighbor_dict(graph) for graph in graphs]
        return graphs_dict

    def _choice_neigbhor(self, rs, index, cur):
        cur_nbrs = self.graph_dicts[index].get(cur)
        assert (cur_nbrs is not None and len(cur_nbrs) > 0
                ), f"{self.algorithm} u-u/u-l graph isolated node {cur}"
        return rs.choice(cur_nbrs)

    def random_walk(self, num_graphs, thresholds, walk_length, start_node, rs):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            probability = rs.random()
            # walk on social graph
            if probability <= thresholds[0]:
                # walk in social graph if current node in social graph
                if cur in self.graph_dicts[0]:
                    walk.append(self._choice_neigbhor(rs, 0, cur))

            else:
                # else walk on second graph(ULG or UCG or ULCG)
                if num_graphs == 2 or (num_graphs == 3
                                       and probability <= thresholds[1]):
                    if cur in self.graph_dicts[1]:
                        walk.append(self._choice_neigbhor(rs, 1, cur))
                else:
                    if cur in self.graph_dicts[2]:
                        walk.append(self._choice_neigbhor(rs, 2, cur))
        return walk

    def _simulate_walks(self, nodes, walk_length, walk_times, thresholds, rs):
        walks = []
        num_graphs = len(self.graph_dicts)
        # parallel walk
        # walked = []
        # with Pool(10) as pool:
        #     for _ in range(walk_times):
        #         rs.shuffle(nodes)
        #         for v in nodes:
        #             walked += [
        #                 pool.apply_async(self.random_walk,
        #                                  args=(num_graphs, thresholds,
        #                                        walk_length, v, rs))
        #             ]
        #     # walks = [walk.get() for walk in walked]

        # single walk
        for _ in range(walk_times):
            rs.shuffle(nodes)
            for v in nodes:
                walk = self.random_walk(num_graphs, thresholds, walk_length, v,
                                        rs)
                walks.append(walk)
        return walks

    def simulate_walks(
        self,
        thresholds,
        seed: int = 0,
    ):
        num_graphs = len(self.graph_dicts)
        assert num_graphs in [2, 3], "check graphs ..."
        nodes = list(self.graph_dicts[0].keys())
        # logging.debug(f'total nodes: {len(nodes)}')
        start = time.time()
        sentences = self._simulate_walks(
            nodes,
            walk_length=self.algorithm_config.walk_length,
            walk_times=self.algorithm_config.walk_times,
            thresholds=thresholds,
            rs=np.random.RandomState(seed),
        )
        if self.algorithm_config.only_user:
            sentences = [[s for s in sen if s in self.graph_dicts[0].keys()]
                         for sen in sentences]
        sentences = [[str(s) for s in sen] for sen in sentences]
        stop = time.time()
        logging.debug(f"{stop-start} seconds walk...")
        return sentences
