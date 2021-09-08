import inspect
import numpy as np
import torch
from torch import nn
from torch.nn import Linear, ReLU, BatchNorm1d, Dropout, Sequential, init
from torch.nn.utils.rnn import PackedSequence
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, \
    accuracy_score, mean_absolute_error, mean_squared_error, \
    median_absolute_error, r2_score, explained_variance_score


class LSTMLayer(nn.Module):
    def __init__(self, atom_fea_len, hidden):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(atom_fea_len, hidden, 1, bias=False)

    def forward(self, atom_in_fea, h, c):
        atom_out_fea, (h, c) = self.lstm(atom_in_fea, (h, c))
        return atom_out_fea, (h, c)

def one_perceptron(in_dim, out_dim, bias=True, activation=ReLU(), bn=False,
                   dropout_rate=None, activation_order='before'):
    perceptron_list = [Linear(in_dim, out_dim, bias)]
    if activation_order == 'before' and activation is not None:
        perceptron_list.append(activation)
    if bn:
        perceptron_list.append(BatchNorm1d(out_dim))
    if activation_order == 'after' and activation is not None:
        perceptron_list.append(activation)
    if dropout_rate is not None:
        perceptron_list.append(Dropout(dropout_rate))
    return perceptron_list


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dims, bias=True, activation=ReLU(),
                 activate_final=True, bn=False, dropout_rate=None,
                 activation_order='before'):
        super(MLP, self).__init__()
        layer_list = list()
        mlp_dims = [input_dim, *output_dims]
        for idx, (in_dim, out_dim) in enumerate(
                zip(mlp_dims[:-1], mlp_dims[1:])):
            if idx == len(output_dims) - 1:
                layer_list.extend(one_perceptron(
                    in_dim, out_dim, bias, activation if activate_final else None,
                    bn, dropout_rate, activation_order))
            else:
                layer_list.extend(one_perceptron(
                    in_dim, out_dim, bias, activation,
                    bn, dropout_rate, activation_order))
        self.mlp = Sequential(*layer_list)

    def forward(self, x):
        return self.mlp(x)


def wrap_packed_sequence(packed_sequence, fn):
    return PackedSequence(
        fn(packed_sequence.data), packed_sequence.batch_sizes,
        packed_sequence.sorted_indices, packed_sequence.unsorted_indices)


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    print('weight init!')
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def summarize_parameters(model):
    summarize_list = list()
    for k, v in model['state_dict'].items():
        print(k, int(np.prod(v.size())))
        summarize_list.append([k, int(np.prod(v.size()))])
    print('total parameters are : {}'.format(
        sum(p.numel() for p in model['state_dict'].values())))
    return summarize_list


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Args:
        prediction(torch.Tensor (N, 1)): Predict tensor.
        target(torch.Tensor (N, 1)): Target tensor.
    """
    return torch.mean(torch.abs(target - prediction))


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def classification_eval(target, prediction):
    """
    Evaluate learning results.

    Args:
        target(torch.Tensor (N, 1)): Target tensor.
        prediction(torch.Tensor (N, 2)): Predict tensor.

    Returns:
        accuracy (float): Accuracy score.
        precision (float): Precision score.
        recall (float): Recall score.
        fscore (float): F score.
        auc_score (float): AUC (Area Under the curve) score.
    """
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    else:
        target = np.array(target)

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.numpy()
    else:
        prediction = np.array(prediction)

    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = precision_recall_fscore_support(
            target_label, pred_label, average='binary')

        auc_score = roc_auc_score(target_label, prediction[:, 1])
        accuracy = accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


def regression_eval(target, prediction):
    mae = mean_absolute_error(target, prediction)
    mse = mean_squared_error(target, prediction)
    mdae = median_absolute_error(target, prediction)
    r2 = r2_score(target, prediction)
    evs = explained_variance_score(target, prediction)
    pcc = pearsonr(target, prediction)[0]
    return mae, mse, mdae, r2, evs, pcc


def adjust_learning_rate(optimizer, factor=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor


def appropriate_kwargs(kwargs, func):
    """
    Auto get the appropriate kwargs according to those allowed by the func.
    Args:
        kwargs (dict): kwargs.
        func (object): function object.

    Returns:
        filtered_dict (dict): filtered kwargs.

    """
    sig = inspect.signature(func)
    filter_keys = [param.name for param in sig.parameters.values()
                   if param.kind == param.POSITIONAL_OR_KEYWORD and
                   param.name in kwargs.keys()]
    appropriate_dict = {filter_key: kwargs[filter_key]
                        for filter_key in filter_keys}
    return appropriate_dict
