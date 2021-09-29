import torch
from torch import nn
from torch.nn import Linear, BatchNorm1d
from torch.nn.utils.rnn import \
    pack_padded_sequence, PackedSequence, pad_packed_sequence
from ggnn.model.utils import MLP


class GNNLayer(nn.Module):
    def __init__(self, node_embedding_len, edge_embedding_len, n_head,
                 attention_len, attention_dropout=0, activation=None,
                 remember_func="residual", device=None):
        """
        Define a GNN layer.
        Args:
            node_embedding_len (int): input and output length of node embedding.
            edge_embedding_len (int):  input and output length of edge embedding.
            n_head (int): number of attention heads implemented.
            attention_len (int):  length of attention vector implemented.
            attention_dropout (float): If non-zero, introduces a Dropout layer
                on the neighbors, with dropout probability equal to dropout.
                Default: 0.
            activation (None or func): activation function. If None, use softplus.
            remember_func (str): "residual" or "lstm".
            device (str): "cuda" or "cpu".
        """
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
            activation_final=False) for _ in range(n_head)])

        self.value_linears = nn.ModuleList([Linear(
            2 * node_embedding_len + edge_embedding_len, attention_len)
            for _ in range(n_head)])

        self.attention_bns = nn.ModuleList(
            [BatchNorm1d(attention_len) for _ in range(n_head)])
        self.attention_softmax = nn.Softmax(dim=1)
        self.attention_drop_layer = nn.Dropout(attention_dropout)

        if n_head * attention_len != node_embedding_len:
            self.after_concat_heads_linear = \
                Linear(n_head * attention_len, node_embedding_len)
            self.after_concat_heads_bn = BatchNorm1d(node_embedding_len)
        else:
            self.after_concat_heads_linear = None
        self.output_bn = BatchNorm1d(node_embedding_len)

        if remember_func == "lstm":
            # self.lstm_func = LSTMLayer(node_embedding_len, node_embedding_len)
            self.lstm_func = nn.LSTM(node_embedding_len, node_embedding_len,
                                     1, bias=False)
            self.lstm_bn = BatchNorm1d(node_embedding_len)

    def forward(self, node_features, edge_features, neighbor_indices,
                neighbor_masks, h=None, c=None):
        """
        Update node_features and edge_features via graph convolution and pooling.
        Most of the complexity arises from the need to deal with different number
        of neighbors for each atom. We use PackedSequence, pad_packed_sequence
        and pack_padded_sequence of torch.nn.utils.rnn to realize the transition.
        Args:
            node_features (Tensor, (batch_size, node_embedding_len)):
            edge_features (Tensor, (batch_size, neighbor_len, edge_embedding_len)):
            neighbor_indices (Tensor, (batch_size, neighbor_len)):
            neighbor_masks (Tensor, (batch_size, neighbor_len)):
            h: for lstm
            c: for lstm
        Returns:
            node_features_updated, edge_features_updated, (h, c)
        """
        batch_len, neighbor_len, _ = edge_features.shape

        # calculate the neighbor length of each atom (in the batch) from the
        # neighbor_masks. In the neighbor_masks with fixed length, "1" means the
        # real neighbor and "0" is for filling the void.
        neighbor_lens = neighbor_masks.sum(dim=1)
        # make the atom with no neighbors in the batch to neighbor of itself?
        neighbor_masks[neighbor_lens == 0] = torch.Tensor(
            [1.] + [0.] * (neighbor_len - 1)).to(self.device)
        neighbor_lens[neighbor_lens == 0] = 1

        # concat node_features, neighbor's node_features, edge_features
        pair_features = torch.cat(
            [node_features.unsqueeze(1).expand(
                batch_len, neighbor_len, self.node_embedding_len),
             node_features[neighbor_indices, :]], dim=2)

        concat_features = torch.cat((pair_features, edge_features), dim=2)

        # change concat_features with fixed length to variable length sequence.
        packed_concat_features = pack_padded_sequence(
            concat_features, neighbor_lens, batch_first=True, enforce_sorted=False)

        # update edge_features and change to fixed length sequence,
        edge_features_updated, _ = pad_packed_sequence(PackedSequence(
            self.edge_bn(self.edge_linear(packed_concat_features.data)),
            packed_concat_features.batch_sizes,
            packed_concat_features.sorted_indices,
            packed_concat_features.unsorted_indices), batch_first=True,
            total_length=neighbor_len)

        # use residual link in the edge feature update
        # edge_features_updated = self.activation(edge_features + padding_tensor(
        #     edge_features_updated, neighbor_len, batch_len, self.device))
        edge_features_updated = self.activation(edge_features +
                                                edge_features_updated)

        # update packed_concat_features
        packed_concat_features = pack_padded_sequence(
            torch.cat((pair_features, edge_features_updated), dim=2),
            neighbor_lens, batch_first=True, enforce_sorted=False)

        # calculate multi-head features for nodes
        head_features_list = list()
        for attention_linear, value_linear, attention_bn in zip(
                self.attention_linears, self.value_linears, self.attention_bns):

            # apply attention_linear to packed_concat_features    
            head_attention, _ = pad_packed_sequence(PackedSequence(
                attention_linear(packed_concat_features.data), 
                packed_concat_features.batch_sizes,
                packed_concat_features.sorted_indices, 
                packed_concat_features.unsorted_indices), batch_first=True,
                total_length=neighbor_len)

            # Masked softmax: calculate the standard softmax and ignore zero values
            masked_attention = head_attention[:, :, -1:].masked_fill(
                    (1 - neighbor_masks.unsqueeze(2)).bool(), float('-inf'))
            head_attention = self.attention_softmax(masked_attention)

            # change head_attention to variable length PackedSequence.
            packed_head_attentions = pack_padded_sequence(
                head_attention, neighbor_lens,
                batch_first=True, enforce_sorted=False)
            
            packed_head_values = PackedSequence(
                value_linear(packed_concat_features.data), 
                packed_concat_features.batch_sizes,
                packed_concat_features.sorted_indices, 
                packed_concat_features.unsorted_indices)

            # head_features tensor
            head_features = self.activation(attention_bn(
                self.attention_drop_layer(packed_head_attentions.data) *
                packed_head_values.data))

            # change head_features to tensor of fixed length
            head_features, _ = pad_packed_sequence(PackedSequence(
                head_features,
                packed_head_attentions.batch_sizes,
                packed_head_attentions.sorted_indices,
                packed_head_attentions.unsorted_indices), batch_first=True,
                total_length=neighbor_len)

            # use sum pooling over neighbors as default
            pooled_head_features = torch.sum(head_features, dim=1)
            head_features_list.append(pooled_head_features)

        # concat multi-head node_features
        concat_heads_features = torch.cat(head_features_list, dim=1)

        # if n_head * attention_len != node_embedding_len
        if self.attention_out_linear is not None:
            node_features_updated = self.output_bn(self.activation(
                self.after_concat_heads_bn(self.after_concat_heads_linear(
                    concat_heads_features))))
        else:
            node_features_updated = self.output_bn(concat_heads_features)

        if self.remember_func == "residual":
            node_features_updated = node_features + node_features_updated
        elif self.remember_func == "lstm":
            node_features_updated, (h, c) = self.lstm_func(
                node_features_updated[None, :], h, c)
            node_features_updated = node_features_updated[0]
            node_features_updated = self.lstm_bn(node_features_updated)
        else:
            raise ValueError("remember_func invalid.")

        return node_features_updated, edge_features_updated, (h, c)
