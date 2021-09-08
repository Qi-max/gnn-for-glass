import torch
import random
import functools
import numpy as np
from typing import List
from torch.utils.data.dataset import IterableDataset
from ggnn.data.batch import GraphBatch
from ggnn.data.graph import GlassGraph
from ggnn.data.sampler import neighbor_sampling
from ggnn.data.utils import list_like


class GraphDatasetIterWrapper(IterableDataset):
    def __init__(self,
                 graph_data_list: List[GlassGraph], device,
                 dataset_indices_list, neighbor_weights, 
                 neighbor_sampling_degree=12, neighbor_masks=None, 
                 batch_size=None, drop_last=False, fill_edge_feature=9,
                 n_hop=4, sampling_method='weight', shuffle=True):
        self.graph_data_list = graph_data_list
        self.neighbor_weights = neighbor_weights
        self.neighbor_masks = neighbor_masks
        self.neighbor_sampling_degree = neighbor_sampling_degree
        self.n_hop = n_hop
        self.device = device
        self.batch_size = batch_size if batch_size != -1 else None
        self.fill_edge_feature = fill_edge_feature
        self.sampling_method = sampling_method
        self.init_iter_indices_(dataset_indices_list, shuffle, drop_last)

    @functools.lru_cache(maxsize=None)  # Cache loaded graph
    def get_graph(self, index):
        return neighbor_sampling(
            self.graph_data_list[index], self.neighbor_weights[index],
            self.neighbor_masks[index], self.neighbor_sampling_degree,
            self.sampling_method)

    def __iter__(self):
        for graph_idx, batch_indices in self.iter_indices:
            mini_batch_indices, mini_batch_hops_indices = None, None
            graph_data, neighbor_masks = self.get_graph(graph_idx)

            if len(graph_data) != len(batch_indices):
                mini_batch_indices = torch.LongTensor(batch_indices).to(self.device)
                graph_batch = GraphBatch(graph_data, neighbor_masks, self.device)
                mini_batch_graph, neighbor_masks, mini_batch_indices, \
                    mini_batch_hops_indices = graph_batch.mini_batch(
                        mini_batch_indices, self.n_hop, self.fill_edge_feature)
            else:
                mini_batch_graph = graph_data

            yield mini_batch_graph, neighbor_masks, mini_batch_hops_indices

    def init_iter_indices_(self, dataset_indices_list, shuffle, drop_last):
        iter_indices = list()
        if isinstance(dataset_indices_list, list_like()):
            if not isinstance(dataset_indices_list[0], list_like()):
                dataset_indices_list = [dataset_indices_list]
        else:
            raise TypeError("Please make sure dataset_indices is list like.")

        if self.batch_size is not None:
            batch_length = 0
            for idx, data_indices in dataset_indices_list:
                if shuffle:
                    shuffle_indices = random.sample(list(data_indices), len(data_indices))
                else:
                    shuffle_indices = data_indices
                if drop_last:
                    last_size = len(shuffle_indices) % self.batch_size
                    batch_num = int((len(shuffle_indices) - last_size) / self.batch_size)
                    batch_length += batch_num
                    iter_indices.extend(list(zip([idx] * batch_num,
                        list(np.array(shuffle_indices[:-last_size]).reshape(
                            batch_num, self.batch_size)))))
                else:
                    now_count = 0
                    now_list = list()
                    for shuffle_idx in shuffle_indices:
                        if now_count == self.batch_size:
                            batch_length += 1
                            iter_indices.append([idx, now_list])
                            now_count = 0
                            now_list = list()
                        now_count += 1
                        now_list.append(shuffle_idx)
                    if now_count > 0:
                        batch_length += 1
                        iter_indices.append([idx, now_list])
            if shuffle:
                random.shuffle(iter_indices)
            self.iter_indices = iter_indices
            self.batch_length = batch_length
        else:
            if shuffle:
                random.shuffle(dataset_indices_list)
            self.iter_indices = dataset_indices_list
            self.batch_length = len(dataset_indices_list)

    def __len__(self):
        return self.batch_length


def dataset_collate(dataset_list):
    """
    Collate a list of small graph and return a large graph.
    :param dataset_list: list of graph data.
    """

    batch_node_features, batch_edge_features, batch_neighbor_indices, \
    batch_neighbor_masks, batch_targets = list(), list(), list(), list(), list()
    batch_hops_indices = None

    base_idx = 0
    for (graph_data, neighbor_masks, mini_batch_hops_indices) in dataset_list:
        if mini_batch_hops_indices is not None:
            if batch_hops_indices is None:
                batch_hops_indices = [list() for _ in mini_batch_hops_indices]
            for i, mini_batch_hop_indices in enumerate(mini_batch_hops_indices):
                batch_hops_indices[i].append(mini_batch_hop_indices)

        batch_node_features.append(graph_data.node_features)
        batch_edge_features.append(graph_data.edge_features)
        batch_neighbor_masks.append(neighbor_masks)
        batch_neighbor_indices.append(graph_data.neighbor_indices + base_idx)
        batch_targets.append(graph_data.targets)
        base_idx += len(graph_data)
    
    mini_batch_hops_indices=[torch.cat(batch_hop_indices, dim=0)
                             for batch_hop_indices in batch_hops_indices]\
                                if batch_hops_indices is not None else None
    neighbor_masks=torch.cat(batch_neighbor_masks, dim=0),
    return GlassGraph(node_features=torch.cat(batch_node_features, dim=0),
                      edge_features=torch.cat(batch_edge_features, dim=0),
                      neighbor_indices=torch.cat(batch_neighbor_indices, dim=0),
                      targets=torch.stack(batch_targets, dim=0)), \
           neighbor_masks, mini_batch_hops_indices
