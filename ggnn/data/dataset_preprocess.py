import torch
import operator as op
import numpy as np
from ggnn.data.graph import GlassGraph
from ggnn.data.utils import GaussianDistance


def build_multi_graphs(df_list, logger, task='regression', neighbor_num=12,
                       train_index=None, val_index=None, test_index=None,
                       neighbor_minus_one=True, start_idx=0, log_target=False,
                       calc_col=None, threshold_respectively=True, operator='ge',
                       threshold_type='quantile', threshold_value=None,
                       target_col='targets', node_feature_col='node_features',
                       neighbor_dist_col='edge_features',
                       neighbor_idx_col='neighbor_indices',
                       type_col='node_type', atom_type='all', fill_radius=9,
                       edge_gaussian_smoothing=False, dmin=0, dmax=8, step=0.2):
    assert task == 'classification' or task == 'regression'
    graph_list = list()

    # calculate classification's target threshold
    if task == 'classification' and calc_col is not None:
        train_threshold, val_threshold, test_threshold = \
            calc_train_val_test_threshold(
                df_list, train_index, val_index, test_index, calc_col,
                threshold_respectively, threshold_type, threshold_value,
                atom_type, type_col)
        logger('train threshold is: {}, \nval threshold is: {}, \ntest threshold'
               ' is: {}'.format(train_threshold, val_threshold, test_threshold))

    for idx, df in enumerate(df_list):
        df.index = np.arange(len(df))

        # build graph node_features
        node_features = torch.Tensor(np.array(list(df[node_feature_col])))

        # build graph's valid node indices
        if atom_type == 'all' or atom_type == -1:
            if 'valid_data' in df.columns:
                node_indices = list(df[df['valid_data']].index)
            else:
                node_indices = list(df.index)
        else:
            if 'valid_data' in df.columns:
                node_indices = list(df[(df[type_col]==atom_type) & (df['valid_data'])].index)
            else:
                node_indices = list(df[(df[type_col]==atom_type)].index)

        # process graph neighbors' ids and distances
        neighbor_dists, neighbor_indices, neighbor_weights, neighbor_masks = \
            list(), list(), list(), list()
        for idx, rows in df[[neighbor_dist_col, neighbor_idx_col]].iterrows():
            neighbor_dist_idx = np.array(list(sorted(zip(*rows[
                [neighbor_dist_col, neighbor_idx_col]].values))))[:neighbor_num]

            neighbor_masks.append([1] * len(neighbor_dist_idx) + 
                                  [0] * (neighbor_num - len(neighbor_dist_idx)))

            neighbor_dist_idx = np.array(
                list(neighbor_dist_idx) + 
                [[fill_radius, 1 if neighbor_minus_one else 0]] *
                (neighbor_num - len(neighbor_dist_idx)))

            neighbor_dists.append([
                dist if dist != 0 else fill_radius
                for dist in neighbor_dist_idx[:, 0].astype(float)])

            neighbor_weights.append([
                (1/dist) if dist !=0 else 0.00001
                for dist in neighbor_dist_idx[:, 0].astype(float)])

            neighbor_indices.append([
                (int(neighbor_id) if int(neighbor_id) != -1 else 0) + start_idx -
                (1 if neighbor_minus_one else 0)
                for neighbor_id in neighbor_dist_idx[:, 1]])

        # build graph edge_features
        if edge_gaussian_smoothing:
            gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step)
            edge_features = torch.Tensor(gdf.expand(np.array(neighbor_dists)))
            fill_edge_feature = torch.Tensor(
                gdf.expand(np.array([fill_radius]))[0])
        else:
            edge_features = torch.Tensor(
                np.expand_dims(np.array(neighbor_dists), axis=-1))
            fill_edge_feature = fill_radius

        # build graph targets
        if task == 'classification':
            if calc_col is not None:
                if idx in train_index:
                    threshold = train_threshold
                elif val_index is not None and idx in val_index:
                    threshold = val_threshold
                elif test_index is not None and idx in test_index:
                    threshold = test_threshold
                df[target_col] = df[calc_col].apply(
                    lambda x: 1 if getattr(op, operator)(x, threshold) else 0)
            targets = torch.LongTensor(np.array(list(df[target_col]))).view(-1, 1)
        else:
            targets = torch.Tensor(np.array(list(df[target_col]))).view(-1, 1)

        # build graph data
        graph_list.append(
            GlassGraph(node_features, edge_features,
                       torch.LongTensor(neighbor_indices),
                       torch.log(targets) if log_target else targets))
    return graph_list, torch.Tensor(neighbor_weights), \
           torch.Tensor(neighbor_masks), node_indices, fill_edge_feature


def calc_train_val_test_threshold(
        df_list, train_index, val_index, test_index, calc_col,
        threshold_respectively, threshold_type, threshold_value,
        atom_type='all', type_col='node_type'):
    val_threshold, test_threshold = None, None
    if train_index is None and val_index is None and test_index is None:
        train_index = list(range(len(df_list)))
    train_df_list = [df_list[train_id] for train_id in train_index]

    if threshold_respectively:
        train_threshold = calc_threshold(
            train_df_list, calc_col, threshold_type,
            threshold_value, atom_type, type_col)

        if val_index is not None:
            val_df_list = [df_list[val_id] for val_id in val_index]
            val_threshold = calc_threshold(
                val_df_list, calc_col, threshold_type,
                threshold_value, atom_type, type_col)

        if test_index is not None:
            test_df_list = [df_list[test_id] for test_id in test_index]
            test_threshold = calc_threshold(
                test_df_list, calc_col, threshold_type,
                threshold_value, atom_type, type_col)
    else:
        train_threshold = calc_threshold(
            df_list, calc_col, threshold_type,
            threshold_value, atom_type, type_col)
        val_threshold, test_threshold = train_threshold, train_threshold

    return train_threshold, val_threshold, test_threshold


def calc_threshold(df_list, calc_col, threshold_type, threshold_value,
                   atom_type='all', type_col='node_type'):
    whole_df = df_list[0][[type_col, calc_col]]
    for df in df_list[1:]:
        whole_df = whole_df.append(df[[type_col, calc_col]], ignore_index=True)

    if atom_type != 'all' and atom_type != -1:
        whole_df = whole_df[whole_df[type_col] == atom_type]

    threshold = whole_df[calc_col].quantile(threshold_value) \
        if threshold_type == 'quantile' else threshold_value
    return threshold
