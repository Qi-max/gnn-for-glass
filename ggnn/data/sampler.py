import torch
from copy import deepcopy
from ggnn.data.graph import GlassGraph


def neighbor_sampling(graph: GlassGraph, neighbor_weights, neighbor_masks=None,
                      neighbor_sampling_degree=12, sampling_method='topk'):
    sampling_graph = deepcopy(graph)
    if sampling_method == 'topk':
        _, neighbor_sampling_idx = torch.topk(
            neighbor_weights, k=neighbor_sampling_degree, dim=1)
    else:
        neighbor_sampling_idx = torch.multinomial(
            neighbor_weights, neighbor_sampling_degree)

    sampling_graph.edge_features = torch.gather(
        sampling_graph.edge_features, 1,
        neighbor_sampling_idx.unsqueeze(-1).expand(
            neighbor_sampling_idx.shape[0],
            neighbor_sampling_idx.shape[1],
            sampling_graph._edge_features.shape[2]))

    sampling_graph.neighbor_indices = torch.gather(
        sampling_graph.neighbor_indices, 1, neighbor_sampling_idx)

    if neighbor_masks is not None:
        neighbor_masks = torch.gather(
            neighbor_masks, 1, neighbor_sampling_idx)
    return sampling_graph, neighbor_masks
