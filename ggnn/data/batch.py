import torch
from copy import deepcopy
from ggnn.data.graph import GlassGraph


class GraphBatch(object):
    def __init__(self, graph: GlassGraph, neighbor_masks=None, device=None):
        self.graph = graph
        self.device = device
        self.neighbor_masks = neighbor_masks

    def mini_batch(self, mini_batch_indices, n_hop, fill_edge_feature=None):
        n_hop_batch_indices = self.get_n_hop_indices(
            torch.LongTensor(mini_batch_indices).to(self.device)
            if self.device else torch.LongTensor(mini_batch_indices), n_hop)

        sub_graph, neighbor_masks, mini_batch_indices = self.get_sub_graph(
            n_hop_batch_indices[0], mini_batch_indices, fill_edge_feature)

        mini_batch_hops_indices = [n_hop_batch_indices[0]]
        for pre_hop_indices, hop_indices in \
                zip(n_hop_batch_indices[:-1], n_hop_batch_indices[1:]):
            re_indices = torch.zeros(pre_hop_indices[-1] + 1).long()
            map_indices = torch.arange(0, len(pre_hop_indices))
            re_indices[pre_hop_indices] = map_indices
            mini_batch_hops_indices.append(re_indices[hop_indices])

        return sub_graph, neighbor_masks, mini_batch_indices, mini_batch_hops_indices

    def get_sub_graph(self, sub_graph_indices, mini_batch_indices, fill_edge_feature):
        re_indices = torch.ones(len(self.graph)).long() * -1
        map_indices = torch.arange(0, len(sub_graph_indices))
        if self.device is not None:
            re_indices = re_indices.to(self.device)
            map_indices = map_indices.to(self.device)
        re_indices[sub_graph_indices] = map_indices
        neighbor_indices = re_indices[self.graph.neighbor_indices[sub_graph_indices]]
        mini_batch_indices = re_indices[mini_batch_indices]

        fill_indices = neighbor_indices == -1
        node_features = self.graph.node_features[sub_graph_indices]
        edge_features = self.graph.edge_features[sub_graph_indices]
        edge_features[fill_indices] = fill_edge_feature

        if self.neighbor_masks:
            neighbor_masks = self.neighbor_masks[sub_graph_indices]
            neighbor_masks[fill_indices] = 1
        else:
            neighbor_masks = None

        targets = self.graph.targets[sub_graph_indices] \
            if self.graph.targets else None
        neighbor_indices[fill_indices] = -1

        sub_graph = GlassGraph(
            node_features, edge_features, neighbor_indices, targets)
        neighbor_masks = self.sort_neighbors_by_id(sub_graph, neighbor_masks)
        return sub_graph, neighbor_masks, mini_batch_indices

    def sort_neighbors_by_id(self, graph, neighbor_masks=None):
        """
        All attributes related to the nearest neighbor are sorted in descending
        order according to the atomic sequence number of the nearest neighbor.

        The purpose of this function is to delete the filled nearest neighbor
        during variable length calculation.
        """
        sort_indices = torch.argsort(
            graph.neighbor_indices, dim=-1, descending=True)
        if self.device is not None:
            sort_indices = sort_indices.to(self.device)

        graph.neighbor_indices = torch.gather(
            graph.neighbor_indices, 1, sort_indices)

        graph.edge_features = torch.gather(
            graph.edge_features, 1, sort_indices.unsqueeze(-1).expand(
                graph.edge_features.shape[0],
                graph.edge_features.shape[1],
                graph.edge_features.shape[2]
            )
        )
        if neighbor_masks:
            neighbor_masks = torch.gather(neighbor_masks, 1, sort_indices)
        return neighbor_masks

    def get_n_hop_indices(self, source_indices, n_hop):
        final_indices = deepcopy(source_indices)
        n_hop_indices = [source_indices]
        for idx, _ in enumerate(range(n_hop)):
            tmp_batch_indices = torch.flatten(
                self.graph.neighbor_indices[final_indices])
            final_indices = torch.unique(
                torch.cat([tmp_batch_indices, final_indices], dim=0),
                sorted=True)
            n_hop_indices.append(final_indices)
        return n_hop_indices[::-1]
