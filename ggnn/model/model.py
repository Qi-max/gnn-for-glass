import torch
from torch import nn
from ggnn.data.graph import GlassGraph
from ggnn.model.layer import GNNLayer
from ggnn.model.utils import MLP


class GNNModel(nn.Module):
    def __init__(self, node_feature_len, node_embedding_lens, edge_embedding_len,
                 n_heads, attention_feature_lens, task="regression", n_class=2,
                 activation=None, in_dropout=0, attention_dropout=0,
                 out_dropout=0, remember_func="residual_link", h_units=None,
                 device=None):
        assert task == "classification" or task == "regression"
        super(GNNModel, self).__init__()
        self.task = task
        self.n_heads = n_heads
        self.h_units = h_units
        self.lstm_hidden = node_embedding_lens[-1]

        self.activation = nn.Softplus() if activation is None else activation
        self.in_drop_layer = nn.Dropout(in_dropout)
        self.node_embedding_layer = MLP(
            node_feature_len, node_embedding_lens, activation=self.activation)
        self.gat_layers = nn.ModuleList([
            GNNLayer(
                node_embedding_len=node_embedding_lens[-1],
                edge_embedding_len=edge_embedding_len, n_head=n_head,
                activation=activation, attention_len=attention_len,
                attention_dropout=attention_dropout,
                remember_func=remember_func, device=self.device)
            for n_head, attention_len in zip(n_heads, attention_feature_lens)])

        # readout
        if self.h_units is not None:
            self.hidden_layers = MLP(
                node_embedding_lens[-1], h_units, activation=self.activation)
        self.out_drop_layer = nn.Dropout(out_dropout)
        self.output_layer = nn.Linear(
            node_embedding_lens[-1] if self.h_units is None else self.h_units[-1],
            n_class if self.task == "classification" else 1)

        if self.task == "classification":
            self.logsoftmax = nn.LogSoftmax(dim=1)

        self.device = device if device is not None else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, graph_data: GlassGraph, neighbor_masks, 
                mini_batch_hops_indices=None):
        node_features = graph_data.node_features
        edge_features = graph_data.edge_features
        neighbor_indices = graph_data.neighbor_indices

        node_features = self.node_embedding_layer(self.in_drop_layer(node_features))
        h = torch.zeros(1, node_features.shape[0], self.lstm_hidden).to(self.device)
        c = torch.zeros(1, node_features.shape[0], self.lstm_hidden).to(self.device)

        if mini_batch_hops_indices is None:
            for gat_layer in self.gat_layers:
                node_features, edge_features, (h, c) = gat_layer(
                    node_features, edge_features, neighbor_indices,
                    neighbor_masks, h, c)
        else:
            # graph mini-batch process
            assert len(self.gat_layers) == len(mini_batch_hops_indices) - 1
            for i, (gat_layer, neighbor_idx_hop) in enumerate(zip(
                    self.gat_layers, mini_batch_hops_indices)):
                if i != 0:
                    re_indices = torch.ones(len(node_features)).long() * -1
                    map_indices = torch.arange(0, len(neighbor_idx_hop))
                    re_indices[neighbor_idx_hop] = map_indices
                    neighbor_indices = re_indices[neighbor_indices[neighbor_idx_hop]]

                    fill_indices = neighbor_indices == -1
                    edge_features = edge_features[neighbor_idx_hop]
                    node_features = node_features[neighbor_idx_hop]
                    neighbor_masks = neighbor_masks[neighbor_idx_hop]
                    edge_features[fill_indices] = 9999
                    neighbor_indices[fill_indices] = 0
                    neighbor_masks[fill_indices] = 0
                    h = h[:, neighbor_idx_hop, :]
                    c = c[:, neighbor_idx_hop, :]

                node_features, edge_features, (h, c) = gat_layer(
                    node_features, edge_features, neighbor_indices,
                    neighbor_masks, h, c)
            node_features = node_features[mini_batch_hops_indices[-1]]

        # readout
        if self.h_units is not None:
            node_features = self.hidden_layers(node_features)

        output = self.output_layer(self.dropout_layer(node_features))
        if self.task == "classification":
            output = self.logsoftmax(output)
        return output
