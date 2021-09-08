import torch
from torch import nn
from torch.nn import Linear, BatchNorm1d
from torch.nn.utils.rnn import \
    pack_padded_sequence, PackedSequence, pad_packed_sequence
from ggnn.data.utils import padding_tensor
from ggnn.model.utils import MLP, LSTMLayer, wrap_packed_sequence


class GNNLayer(nn.Module):
    def __init__(self, node_embedding_len, edge_embedding_len, n_head,
                 attention_len, attention_dropout=0, activation=None,
                 remember_func="residual_link", device=None):
        super(GNNLayer, self).__init__()
        self.node_embedding_len = node_embedding_len
        self.edge_embedding_len = edge_embedding_len
        self.attention_len = attention_len
        self.remember_func = remember_func
        self.activation = nn.Softplus() if activation is None else activation
        self.device = device

        self.edge_linear = Linear(
            2 * node_embedding_len + edge_embedding_len, edge_embedding_len)
        self.edge_bn = BatchNorm1d(edge_embedding_len)

        self.attention_linears = nn.ModuleList([MLP(
            2 * node_embedding_len + edge_embedding_len, [attention_len, 1], 
            activate_final=False) for _ in range(n_head)])

        self.value_linears = nn.ModuleList([Linear(
            2 * node_embedding_len + edge_embedding_len, attention_len)
            for _ in range(n_head)])

        self.attention_bns = nn.ModuleList(
            [BatchNorm1d(attention_len) for _ in range(n_head)])
        self.attention_softmax = nn.Softmax(dim=1)
        self.attention_drop_layer = nn.Dropout(attention_dropout)

        if n_head * attention_len != node_embedding_len:
            self.attention_out_linear = \
                Linear(n_head * attention_len, node_embedding_len)
            self.attention_out_bn = BatchNorm1d(node_embedding_len)
        else:
            self.attention_out_linear = None
        self.output_bn = BatchNorm1d(node_embedding_len)

        if remember_func == "lstm":
            self.lstm_func = LSTMLayer(node_embedding_len, node_embedding_len)
            self.lstm_bn = BatchNorm1d(node_embedding_len)

    def forward(self, node_features, edge_features, neighbor_indices,
                neighbor_masks, h, c):
        heads_node_features = list()
        batch_len, neighbor_len, _ = edge_features.shape

        # calculate the sequence lengths of each batch element from masks.
        neighbor_lens = neighbor_masks.sum(dim=1)
        neighbor_masks[neighbor_lens == 0] = torch.Tensor(
            [1.] + [0.] * (neighbor_len - 1)).to(self.device)
        neighbor_lens[neighbor_lens == 0] = 1

        # concat node_features, neighbor's node_features, edge_features
        pair_node_feature = torch.cat(
            [node_features.unsqueeze(1).expand(
                batch_len, neighbor_len, self.node_embedding_len),
             node_features[neighbor_indices, :]], dim=2)
        total_feature = torch.cat((pair_node_feature, edge_features), dim=2)

        # change total_features to variable length sequence.
        packed_total_feature = pack_padded_sequence(
            total_feature, neighbor_lens, batch_first=True, enforce_sorted=False)

        # update edge_features and change it to fixed length sequence
        edge_feature_out, _ = pad_packed_sequence(PackedSequence(
            self.edge_bn(self.edge_linear(packed_total_feature.data)),
            packed_total_feature.batch_sizes, packed_total_feature.sorted_indices,
            packed_total_feature.unsorted_indices), batch_first=True)
        edge_feature_out = self.activation(edge_features + padding_tensor(
            edge_feature_out, neighbor_len, batch_len, self.device))

        # change updated total_feature to variable length sequence.
        packed_total_feature = pack_padded_sequence(
            torch.cat((pair_node_feature, edge_feature_out), dim=2),
            neighbor_lens, batch_first=True, enforce_sorted=False)

        # calculate multi-head node_features
        for attention_linear, value_linear, attention_bn in zip(
                self.attention_linears, self.value_linears, self.attention_bns):
            head_attention, _ = pad_packed_sequence(wrap_packed_sequence(
                packed_total_feature, attention_linear), batch_first=True)

            # Masked softmax: calculate the standard softmax and ignore zero values
            masked_attention = padding_tensor(
                head_attention, neighbor_len, batch_len,
                self.device)[:, :, -1:].masked_fill(
                    (1 - neighbor_masks.unsqueeze(2)).bool(), float('-inf'))
            head_attention = self.attention_softmax(masked_attention)

            # change head_attention to variable length sequence.
            packed_head_att_feature = pack_padded_sequence(
                head_attention, neighbor_lens,
                batch_first=True, enforce_sorted=False)

            packed_value_data = wrap_packed_sequence(
                packed_total_feature, value_linear)

            head_node_feature_data = self.activation(attention_bn(
                self.attention_drop_layer(packed_head_att_feature.data) *
                packed_value_data.data))

            # change updated node_features to fixed length sequence
            head_node_feature, _ = pad_packed_sequence(PackedSequence(
                head_node_feature_data, packed_head_att_feature.batch_sizes,
                packed_head_att_feature.sorted_indices,
                packed_head_att_feature.unsorted_indices), batch_first=True)
            head_node_feature = torch.sum(padding_tensor(
                head_node_feature, neighbor_len, batch_len, self.device), dim=1)
            heads_node_features.append(head_node_feature)

        # concat multi-head node_features
        heads_node_features = torch.cat(heads_node_features, dim=1)

        if self.attention_out_linear is not None:
            updated_node_features = self.output_bn(self.activation(
                self.attention_out_bn(
                    self.attention_out_linear(heads_node_features))))
        else:
            updated_node_features = self.output_bn(heads_node_features)

        if self.remember_func == "residual_link":
            atom_out_feature = node_features + updated_node_features
        elif self.remember_func == "lstm":
            atom_conv_feature = updated_node_features[None, :]
            atom_out_feature, (h, c) = self.lstm_func(atom_conv_feature, h, c)
            atom_out_feature = atom_out_feature[0]
            atom_out_feature = self.lstm_bn(atom_out_feature)
        else:
            raise ValueError("remember_func invalid.")

        return atom_out_feature, edge_feature_out, (h, c)
