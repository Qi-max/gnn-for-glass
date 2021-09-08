import os
import csv
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from ggnn.model.utils import \
    adjust_learning_rate, classification_eval, regression_eval


@torch.no_grad()
def predict_step(data_loader:DataLoader, model, loss_func, optimizer,
                 normalizer, logger, output_path=None, test=False, task_tag=None,
                 task='regression', print_freq=10, save_test=False, device='cuda'):

    start_time = time.time()
    with torch.no_grad():
        task_tag = ("Test" if test else "Validate") if task_tag is None else task_tag
        losses, batch_times, total_nodes = 0, 0, 0
        total_targets, total_preds, total_graph_indices = list(), list(), list()

        # switch to evaluate mode
        model.eval()

        for i, (graph_data, neighbor_masks, mini_batch_hops_indices) in \
                enumerate(data_loader):
            total_nodes += graph_data.targets.size(0)
            graph_data.to(device)
            neighbor_masks.to(device)
            mini_batch_hops_indices.to(device)

            targets_normed = normalizer.norm(graph_data.targets).view(-1, 1) \
                if task == 'regression' else graph_data.targets.view(-1).long()
            targets_normed = targets_normed.to(device)

            output = model(graph_data, neighbor_masks, mini_batch_hops_indices)
            loss = loss_func(output, targets_normed)

            if torch.isnan(loss) or torch.isinf(loss):
                now_lr = optimizer.param_groups[0]['lr']
                adjust_learning_rate(optimizer, factor=0.2)
                logger("adjust learning rate from {} to {}".format(
                    now_lr, optimizer.param_groups[0]['lr']))
                return None

            # measure accuracy and record loss
            losses += loss.item() * targets_normed.size(0)
            total_target = graph_data.targets.view(-1)
            if task == 'regression':
                total_preds += normalizer.denorm(output.cpu()).view(-1).tolist()
                total_targets += total_target.tolist()
            else:
                if targets_normed.view(-1).shape[0] == output.shape[0]:
                    total_pred = torch.exp(output.cpu())
                else:
                    total_pred = torch.exp(output.cpu())
                assert total_pred.shape[1] == 2
                total_preds += total_pred.tolist()
                total_targets += total_target.tolist()

            # measure elapsed time
            batch_time = time.time() - start_time
            batch_times += batch_time * targets_normed.size(0)

            if i % print_freq == 0:
                logger('Epoch: {0}: [{1}/{2}]\tData Size: {3}\t'
                       'Time {4:.3f} ({5:.3f})\tLoss {6:.4f} ({7:.4f})'.format(
                    task_tag, i, len(data_loader) if not isinstance(
                        data_loader.dataset, IterableDataset) else 'N',
                    output.data.shape[0], batch_time, batch_times / total_nodes,
                    loss.item(), losses/total_nodes))

        if test and save_test:
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, 'test_results.csv'), 'w+',
                      newline='') as f:
                writer = csv.writer(f)
                for graph_indices, target, pred in zip(
                        total_graph_indices, total_targets, total_preds):
                    writer.writerow((graph_indices, target, pred))

        if task == 'regression':
            predict_mae, predict_mse, predict_mdae, \
            predict_r2, predict_evs, pcc = regression_eval(
                total_targets, total_preds)
            logger('Data Size: {0}\tTime {1:.4f}\tMAE {2:.4f}\t'
                   'MSE {3:.4f}\tMDAE {4:.4f}\tR2 {5:.4f}\t'
                   'EVS {6:.4f}\tPearsonCorrelationCoefficient {7:.4f}'.format(
                len(data_loader) if not isinstance(
                    data_loader.dataset, IterableDataset) else 'N',
                batch_time, predict_mae, predict_mse,
                predict_mdae, predict_r2, predict_evs, pcc))
            return (predict_mae, predict_mse, predict_mdae, predict_r2, predict_evs, pcc)
        else:
            predict_accuracy, predict_precision, predict_recall, \
            predict_fscore, predict_auc_score = classification_eval(
                total_targets, total_preds)
            logger('Data Size: {0}\tUseTime {1:.4f}\tAccuracy {2:.4f}\t'
                   'Precision {3:.4f}\tRecall {4:.4f}\tF1 {5:.4f}\t'
                   'AUC {6:.4f}'.format(
                len(data_loader) if not isinstance(
                    data_loader.dataset, IterableDataset) else 'N',
                batch_time, predict_accuracy, predict_precision,
                predict_recall, predict_fscore, predict_auc_score))
            return (predict_accuracy, predict_precision, predict_recall,
                    predict_fscore, predict_auc_score)
