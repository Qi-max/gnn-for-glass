import time
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from ggnn.model.utils import regression_eval, classification_eval


def train_step(train_loader: DataLoader, model, loss_func, optimizer,
               epoch, normalizer, task='regression', print_freq=1, logger=None,
               measure_metrics="auc", train_score_simpler=False, device='cuda'):

    assert task == "classification" or task == "regression"
    total_nodes = 0
    losses = 0
    batch_times = 0
    data_loader_times = 0
    total_targets = []
    total_preds = []

    # switch to train mode
    model.train()
    start_time = time.time()

    for i, (graph_data, neighbor_masks, mini_batch_hops_indices) in \
            enumerate(train_loader):
        data_loader_time = time.time() - start_time
        data_loader_times += data_loader_time * graph_data.targets.size(0)
        total_nodes += graph_data.targets.size(0)

        graph_data.to(device)
        neighbor_masks.to(device)
        mini_batch_hops_indices.to(device)

        # normalize target
        targets_normed = normalizer.norm(deepcopy(graph_data.targets)).view(-1, 1) \
            if task == 'regression' else graph_data.targets.view(-1).long()
        targets_normed = targets_normed.to(device)

        output = model(graph_data, neighbor_masks, mini_batch_hops_indices)
        loss = loss_func(output, targets_normed)

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError('loss is nan or inf.')

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item() * targets_normed.size(0)
        if train_score_simpler:
            if task == 'regression':
                total_pred = normalizer.denorm(output.cpu())
                total_preds += total_pred.view(-1).tolist()
            else:
                total_pred = torch.exp(output.cpu())
                total_preds += total_pred.tolist()
            total_targets += graph_data.targets.view(-1)[
                graph_data.origin_graph_indices].tolist()

        # measure elapsed time
        batch_time = time.time() - start_time
        batch_times += batch_time * targets_normed.size(0)

        if i % print_freq == 0:
            if task == 'regression':
                logger('Epoch: [{0}][{1}/{2}]\tSubGraph Size: {3}\t'
                       'Batch_Time {4:.3f} ({5:.3f})\t'
                       'Loader_Time {7:.3f} ({7:.3f})\t'
                       'Loss {8:.4f} ({9:.4f})'.format(
                    epoch, i, len(train_loader) if not
                    isinstance(train_loader.dataset, IterableDataset) else 'N',
                    targets_normed.shape[0], batch_time,
                    batch_times / total_nodes, data_loader_time,
                    data_loader_times / total_nodes,
                    loss.item(), losses / total_nodes))
            else:
                logger('Epoch: [{0}][{1}/{2}]\t'
                       'SubGraph Size: {3}\t'
                       'Batch_Time {4:.3f} ({5:.3f})\t'
                       'Loader_Time {6:.3f} ({7:.3f})\t'
                       'Loss {8:.4f} ({9:.4f})'.format(
                    epoch, i, len(train_loader) if not
                    isinstance(train_loader.dataset, IterableDataset) else 'N',
                    targets_normed.shape[0], batch_time,
                    batch_times / total_nodes, data_loader_time,
                    data_loader_times / total_nodes,
                    loss.item(), losses / total_nodes))

    if train_score_simpler:
        if task == 'regression':
            mae, mse, mdae, r2, evs, pcc = \
                regression_eval(total_targets, total_preds)
            logger('MAE {0:.6f}\tMSE {1:.6f}\tMDAE {2:.6f}\tR2 {3:.6f}\t'
                   'EVS {4:.6f}\tPCC {5:.6f}'.format(mae, mse, mdae, r2, evs, pcc))
            return -r2 if measure_metrics=="r2" else mae, losses / total_nodes
        else:
            accuracy, precision, recall, fscore, auc_score = \
                classification_eval(total_targets, total_preds)
            logger('Accuracy {0:.6f}\tPrecision {1:.6f}\tRecall {2:.6f}\t'
                   'F1 {3:.6f}\tAUC {4:.6f}'.format(
                accuracy, precision, recall, fscore, auc_score))
            return auc_score if measure_metrics=="auc" \
                       else (auc_score + fscore)/2, losses / total_nodes
    return -1, losses / total_nodes
