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


_attention_dict = {"relu": nn.ReLU()}


def single_layer_perceptron(in_dim, out_dim, bias=True, activation=ReLU(),
                            bn=True, dropout=0, activation_before_bn=True):
    """
    Build a single layer perceptron model.
    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        bias (bool): whether to add bias in the linear transformation.
        activation (func): activation function.
        bn (bool): whether to perform batch-normalization.
        dropout (float): If non-zero, introduces a Dropout layer on the output,
            with dropout probability equal to dropout. Default: 0
        activation_before_bn (bool): whether to perform activation before
            batch normalization. Default: True.
    Returns:
        modules (list)
    """
    modules = [Linear(in_dim, out_dim, bias)]
    if activation_before_bn == 'before' and activation is not None:
        modules.append(activation)
    if bn:
        modules.append(BatchNorm1d(out_dim))
    if activation_before_bn == False and activation is not None:
        modules.append(activation)
    if dropout > 0:
        modules.append(Dropout(dropout))
    return modules


class MLP(torch.nn.Module):
    """
    Build a multi-layer perception (MLP) model.
    """
    def __init__(self, input_dim, output_dims, bias=True, activation=ReLU(),
                 activation_final=ReLU(), bn=True, dropout=0,
                 activation_before_bn=True):
        """
        Create a mlp model.
        Args:
            input_dim (int): input dimension.
            output_dims (list): a list of output dimensions. The length
                corresponds to the number of layers.
            bias (bool): whether to add bias in the linear transformation.
            activation (func): activation function in the middle layers.
            activation_final (func or None): activation function in the output
                layer. If set to None, no activation function will be applied.
            bn (bool): whether to perform batch-normalization.
            dropout (float): If non-zero, introduces a Dropout layer on the
                outputs of each layer, with dropout probability equal to
                dropout. Default: 0.
            activation_before_bn (bool): whether to perform activation before
                batch normalization. Default: True.
        """
        super(MLP, self).__init__()
        modules = list()
        mlp_dims = [input_dim, *output_dims]
        for idx, (in_dim, out_dim) in enumerate(
                zip(mlp_dims[:-1], mlp_dims[1:])):
            if idx == len(output_dims) - 1:
                modules.extend(single_layer_perceptron(
                    in_dim, out_dim, bias, activation if activation_final else None,
                    bn, dropout, activation_before_bn))
            else:
                modules.extend(single_layer_perceptron(
                    in_dim, out_dim, bias, activation,
                    bn, dropout, activation_before_bn))
        self.mlp = Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)


def weight_init(m):
    '''
    Usage:
        model = Model()  # Instantiate a model
        model.apply(weight_init)
    '''
    print('weight initialization!')
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


def summarize_model_parameters(model):
    summary_list = list()
    for k, v in model.state_dict().items():
        print(k, int(np.prod(v.size())))
        summary_list.append([k, int(np.prod(v.size()))])
    print('total parameters are : {}'.format(
        sum(p.numel() for p in model.state_dict().values())))
    return summary_list


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target
    Args:
        prediction(Tensor (N, 1)): Prediction tensor.
        target(Tensor (N, 1)): Target tensor.
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
    Evaluate classification results and return several commonly-used metrics.
    Args:
        target(Tensor (N, 1)): Target tensor.
        prediction(Tensor (N, 2)): Predict tensor.
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
    """
    Evaluate regression results and return several commonly-used metrics.
    Args:
        target(Tensor (N, 1)): Target tensor.
        prediction(Tensor (N, 1)): Predict tensor.
    Returns:
        mae (float): Mean absolute error score.
        mse (float): Mean squared error score.
        mdae (float): Median absolute error score.
        r2 (float): R2 score.
        evs (float): Explained variance score.
        pcc (float): Pearson correlation coefficient.
    """
    mae = mean_absolute_error(target, prediction)
    mse = mean_squared_error(target, prediction)
    mdae = median_absolute_error(target, prediction)
    r2 = r2_score(target, prediction)
    evs = explained_variance_score(target, prediction)
    pcc = pearsonr(target, prediction)[0]
    return mae, mse, mdae, r2, evs, pcc


def adjust_learning_rate(optimizer, factor=0.1):
    """Sets the learning rate to the original value * factor"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor


def appropriate_kwargs(kwargs, func):
    """
    Filter the appropriate kwargs that are allowed by the function.
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
    filtered_dict = {filter_key: kwargs[filter_key]
                     for filter_key in filter_keys}
    return filtered_dict
