import torch
from torch import nn
from ggnn.data.graph import GlassGraph
from ggnn.model.layer import GNNLayer
from ggnn.model.utils import MLP


class GNNModel(nn.Module):
    def __init__(self, node_feature_len, edge_embedding_len,
                 init_node_embedding_units,
                 n_heads=4, attention_feature_lens=(),
                 task="regression", n_class=2, activation=None,
                 remember_func="residual",
                 in_dropout=0, attention_dropout=0, readout_dropout=0,
                 readout_hidden_units=None,
                 device=None):

        assert task == "classification" or task == "regression"
        super(GNNModel, self).__init__()
        self.task = task
        self.n_heads = n_heads
        self.readout_hidden_units = readout_hidden_units
        self.remember_func = remember_func
        # self.lstm_hidden = init_node_embedding_units[-1]

        self.activation = nn.Softplus() if activation is None else activation
        self.input_dropout_layer = nn.Dropout(in_dropout)

        # force the node embedding to be of node_feature_len
        if init_node_embedding_units[-1] != node_feature_len:
            init_node_embedding_units = list(init_node_embedding_units[:-1]) + \
                                       [node_feature_len]
        self.node_embedding_layer = MLP(
            node_feature_len, init_node_embedding_units,
            activation=self.activation)

        self.gnn_layers = nn.ModuleList([
            GNNLayer(
                node_embedding_len=node_feature_len,
                edge_embedding_len=edge_embedding_len, n_head=n_head,
                activation=activation, attention_len=attention_len,
                attention_dropout=attention_dropout,
                remember_func=remember_func, device=self.device)
            for n_head, attention_len in zip(n_heads, attention_feature_lens)])

        # readout
        if self.readout_hidden_units is not None:
            self.readout_hidden_layers = MLP(
                node_feature_len, self.readout_hidden_units,
                activation=self.activation)
            self.readout_layer = nn.Linear(
                self.readout_hidden_units[-1], n_class
                if self.task == "classification" else 1)
        else:
            self.readout_layer = nn.Linear(
                node_feature_len, n_class
                if self.task == "classification" else 1)
        self.readout_dropout_layer = nn.Dropout(readout_dropout)

        if self.task == "classification":
            self.logsoftmax = nn.LogSoftmax(dim=1)

        self.device = device if device is not None else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, graph_data: GlassGraph, neighbor_masks, 
                mini_batch_hops_indices=None):
        node_features = graph_data.node_features
        edge_features = graph_data.edge_features
        neighbor_indices = graph_data.neighbor_indices

        node_features = self.node_embedding_layer(
            self.input_dropout_layer(node_features))

        if self.remember_func == "lstm":
            h = torch.zeros(
                1, node_features.shape[0], self.lstm_hidden).to(self.device)
            c = torch.zeros(
                1, node_features.shape[0], self.lstm_hidden).to(self.device)
        else:
            h, c = None, None

        if mini_batch_hops_indices is None:
            for gnn_layer in self.gnn_layers:
                node_features, edge_features, (h, c) = gnn_layer(
                    node_features, edge_features, neighbor_indices,
                    neighbor_masks, h, c)
        else:
            # graph mini-batch process
            assert len(self.gnn_layers) == len(mini_batch_hops_indices) - 1
            for i, (gnn_layer, neighbor_idx_hop) in enumerate(zip(
                    self.gnn_layers, mini_batch_hops_indices)):
                if i != 0:
                    re_indices = torch.ones(len(node_features)).long() * - 1
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

                node_features, edge_features, (h, c) = gnn_layer(
                    node_features, edge_features, neighbor_indices,
                    neighbor_masks, h, c)
            node_features = node_features[mini_batch_hops_indices[-1]]

        # readout
        if self.readout_hidden_units is not None:
            node_features = self.hidden_layers(node_features)

        predictions = self.readout_layer(self.dropout_layer(node_features))
        if self.task == "classification":
            predictions = self.logsoftmax(predictions)
        return predictions
