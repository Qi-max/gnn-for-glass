import os
import time
import warnings
import torch
import numpy as np
from copy import deepcopy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from ggnn.predict import predict_step
from ggnn.logger import Logger
from ggnn.train import train_step


class GNNLearn(object):
    def __init__(self, train_dataset_wrapper, graph_model, target_normalizer, optimizer,
                 loss_func, scheduler, task='regression', disable_cuda=False,
                 warm_start_file=None, warm_start_best=False, output_path=".",
                 save_best=True, save_last=False, checkpoint_interval=None,
                 measure_metrics='auc', eval_train_dataset_wrapper=None,
                 val_dataset_wrapper=None, test_dataset_wrapper=None,
                 train_score_simpler=False, device=None):
        self.train_dataset_wrapper = train_dataset_wrapper
        self.val_dataset_wrapper = val_dataset_wrapper
        self.test_dataset_wrapper = test_dataset_wrapper
        self.eval_train_dataset_wrapper = eval_train_dataset_wrapper
        self.graph_model = graph_model
        self.target_normalizer = target_normalizer
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.scheduler = scheduler
        self.task = task
        self.warm_start_file = warm_start_file
        self.warm_start_best = warm_start_best
        self.disable_cuda = disable_cuda
        self.save_best = save_best
        self.save_last = save_last
        self.checkpoint_interval = checkpoint_interval
        self.output_path = output_path
        self.measure_metrics = measure_metrics
        self.train_score_simpler = train_score_simpler
        self.device = device if device is not None else torch.device(
            'cuda' if not self.disable_cuda and torch.cuda.is_available() else 'cpu')

        os.makedirs(self.output_path, exist_ok=True)
        self.simple_logger = Logger(os.path.join(
            self.output_path, 'simple_logger.log'))
        self.best_logger = Logger(os.path.join(
            self.output_path, 'best_logger.log'))
        self.detail_logger = Logger(os.path.join(
            self.output_path, 'detail_logger.log'))

    def fit(self, fit_type, **kwargs):
        """
        Args:
            fit_type: String
                Use inductive or transductive learning
        Returns:
        """
        if self.output_path is not None and kwargs.get('max_epochs', 1000) > 0:
            os.makedirs(os.path.join(self.output_path, 'model'), exist_ok=True)
            self.last_file = os.path.join(
                self.output_path, 'model', 'last.pth.tar')
            self.checkpoint_file = os.path.join(
                self.output_path, 'model',
                'checkpoint_best_epoch_{:08}_score_{}_now_epoch_{}.pth.tar')
            self.best_model_file = os.path.join(
                self.output_path, 'model',
                'best_model_epoch_{:08}_score_{}.pth.tar')

        if self.warm_start_file is not None:
            start_epoch, best_epoch, best_score = self._warm_start()
        else:
            start_epoch, best_epoch = 0, 0
            self._best_model = deepcopy(self.graph_model)
            best_score = 0. if self.task == 'classification' else 1e10
        self.start_epoch = start_epoch

        self.graph_model.to(self.device)
        if fit_type == 'inductive':
            self._inductive_fit(
                start_epoch, best_epoch, best_score, **kwargs)
        return self

    def predict(self, inductive=False, **kwargs):
        if inductive == 'inductive':
            self._inductive_predict(**kwargs)
        return

    def _inductive_fit(self, start_epoch, best_epoch, best_score,
                       print_freq=1, test=False, **kwargs):
        self._latest_model = self.graph_model

        start_time = time.time()
        if self.val_dataset_wrapper is not None:
            val_loader = DataLoader(
                self.val_dataset_wrapper,
                worker_init_fn=kwargs.get('worker_init_fn', None),
                num_workers=kwargs.get('num_workers', 0),
                collate_fn=kwargs.get('collate_fn', None),
                pin_memory=kwargs.get('pin_memory', False))

        if self.test_dataset_wrapper is not None:
            test_loader = DataLoader(
                self.test_dataset_wrapper,
                worker_init_fn=kwargs.get('worker_init_fn', None),
                num_workers=kwargs.get('num_workers', 0),
                collate_fn=kwargs.get('collate_fn', None),
                pin_memory=kwargs.get('pin_memory', False))

        if self.eval_train_dataset_wrapper is not None:
            eval_train_loader = DataLoader(
                self.eval_train_dataset_wrapper,
                worker_init_fn=kwargs.get('worker_init_fn', None),
                num_workers=kwargs.get('num_workers', 0),
                collate_fn=kwargs.get('collate_fn', None),
                pin_memory=kwargs.get('pin_memory', False))

        print('Initialize data loader time: {}'.format(time.time() - start_time))

        if self.train_dataset_wrapper.sampling_method == 'topk':
            train_loader = DataLoader(
                self.train_dataset_wrapper,
                worker_init_fn=kwargs.get('worker_init_fn', None),
                num_workers=kwargs.get('num_workers', 0),
                collate_fn=kwargs.get('collate_fn', None),
                pin_memory=kwargs.get('pin_memory', False))

        for epoch in range(start_epoch, kwargs.get('max_epochs', 1000)):
            if self.train_dataset_wrapper.sampling_method != 'topk':
                # reset train loader's neighbor sampling of each epoch
                self.train_dataset_wrapper.get_structure_data.cache_clear()
                train_loader = DataLoader(
                    self.train_dataset_wrapper,
                    worker_init_fn=kwargs.get('worker_init_fn', None),
                    num_workers=kwargs.get('num_workers', 0),
                    collate_fn=kwargs.get('collate_fn', None),
                    pin_memory=kwargs.get('pin_memory', False))

            train_score, loss_value, validate_score, test_score = \
                None, None, None, None
            while train_score is None or loss_value is None or \
                    validate_score is None or test_score is None:
                train_score, loss_value = train_step(
                    train_loader=train_loader, model=self.graph_model,
                    loss_func=self.loss_func, optimizer=self.optimizer,
                    epoch=epoch, target_normalizer=self.target_normalizer,
                    logger=self.detail_logger, task=self.task,
                    device=self.device, print_freq=print_freq,
                    measure_metrics=self.measure_metrics,
                    train_score_simpler=self.train_score_simpler)
                if loss_value is None:
                    warnings.warn("epoch {}: loss is None".format(epoch))
                    continue

                if not self.train_score_simpler:
                    train_score = predict_step(
                        data_loader=train_loader
                        if self.eval_train_dataset_wrapper is None
                        else eval_train_loader,
                        model=self.graph_model, loss_func=self.loss_func,
                        optimizer=self.optimizer, target_normalizer=self.target_normalizer,
                        test=False, logger=self.detail_logger, task=self.task,
                        device=self.device, print_freq=print_freq,
                        measure_metrics=self.measure_metrics, task_tag='Train')

                if train_score is None:
                    warnings.warn("epoch {}: train_score is None".format(epoch))
                    continue

                if self.val_dataset_wrapper is not None:
                    validate_score = predict_step(
                        data_loader=val_loader, model=self.graph_model,
                        loss_func=self.loss_func, optimizer=self.optimizer,
                        target_normalizer=self.target_normalizer, test=False,
                        logger=self.detail_logger, task=self.task,
                        device=self.device, print_freq=print_freq,
                        output_path=self.output_path,
                        measure_metrics=self.measure_metrics)
                else:
                    validate_score = -1

                if validate_score is None:
                    warnings.warn("epoch {}: validate_score is None".format(epoch))
                    continue

                if test:
                    if epoch == kwargs.get('max_epochs', 1000) - 1:
                        self.save_test = True

                    test_score = predict_step(
                        data_loader=test_loader, model=self.graph_model,
                        loss_func=self.loss_func, optimizer=self.optimizer,
                        target_normalizer=self.target_normalizer, test=test,
                        logger=self.detail_logger, task=self.task,
                        device=self.device, print_freq=print_freq,
                        output_path=self.output_path,
                        measure_metrics=self.measure_metrics)
                else:
                    test_score = -1

            self.simple_logger("{},{},{},{},{}".format(epoch, loss_value,
                                                       train_score,
                                                       validate_score,
                                                       test_score))

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(validate_score)
                print('Now learning rate is: {}.'.format(
                    self.optimizer.param_groups[0]['lr']))
                self.detail_logger('Now learning rate is: {}.'.format(
                    self.optimizer.param_groups[0]['lr']))
            else:
                self.scheduler.step()

            # Calculate best score
            if self.task == 'regression':
                is_best = validate_score < best_score
                best_score = min(validate_score, best_score)
            else:
                is_best = validate_score > best_score
                best_score = max(validate_score, best_score)

            if is_best:
                self.best_logger("{},{},{},{},{}".format(
                    epoch, loss_value, train_score, validate_score, test_score))

                self._best_model, best_epoch = deepcopy(self.graph_model), epoch
                if self.output_path is not None:
                    self._save_model(
                        epoch, best_epoch, best_score, self.optimizer,
                        self.target_normalizer, self.scheduler, self.best_model_file)
            self._latest_model = self.graph_model

            # Save checkpoint
            if self.checkpoint_interval is not None and \
                    (self.output_path is not None and
                     epoch % self.checkpoint_interval == 0):
                self._save_model(
                    epoch, best_epoch, best_score, self.optimizer,
                    self.target_normalizer, self.scheduler, self.checkpoint_file)

            elif self.save_last and self.output_path is not None:
                self._save_model(
                    epoch, best_epoch, best_score, self.optimizer,
                    self.target_normalizer, self.scheduler, self.last_file)

        return self

    def _inductive_predict(self, num_workers=0, collate_fn=None,
                           pin_memory=False, sampling='normal'):
        self.graph_model.to(self.device)

        val_loader = DataLoader(
            dataset=self.val_dataset_wrapper,
            collate_fn=default_collate if collate_fn is None else collate_fn,
            num_workers=num_workers, pin_memory=pin_memory)

        if self.test_dataset_wrapper is not None:
            test_loader = DataLoader(
                self.test_dataset_wrapper,
                collate_fn=default_collate if collate_fn is None else collate_fn,
                num_workers=num_workers, pin_memory=pin_memory)

        self._latest_model = self.graph_model
        validate_score = predict_step(
            data_loader=val_loader, model=self.graph_model,
            loss_func=self.loss_func, target_normalizer=self.target_normalizer,
            optimizer=self.optimizer, test=True, save_test=True,
            logger=self.detail_logger, task=self.task, device=self.device,
            print_freq=1, output_path=self.output_path, all_metrics=True,
            sampling=sampling)

        test_score = -1
        if self.test_dataset_wrapper is not None:
            test_score = predict_step(
                data_loader=test_loader, model=self.graph_model,
                loss_func=self.loss_func, target_normalizer=self.target_normalizer,
                optimizer=self.optimizer, test=True, save_test=True,
                logger=self.detail_logger, task=self.task, device=self.device,
                print_freq=1, output_path=self.output_path, all_metrics=True,
                sampling=sampling)

        self.simple_logger("{},{},{}".format(
            self.start_epoch, validate_score, test_score))

        if validate_score is np.nan:
            raise ValueError("Exit due to predict score is NaN")
        return self

    def _warm_start(self):
        if not os.path.exists(self.warm_start_file):
            raise FileNotFoundError("Warm start file not found.")
        else:
            if os.path.isfile(self.warm_start_file):
                warm_start_file = self.warm_start_file
            else:
                if self.warm_start_best:
                    file_list = [file for file in os.listdir(self.warm_start_file)
                                 if file.startswith('best_model_')]
                else:
                    file_list = [file for file in os.listdir(self.warm_start_file)
                                 if file.startswith('checkpoint_')]

                reverse = False if self.task == 'regression' and \
                                   self.warm_start_best else True
                sorted_key = \
                    lambda x: float(x.split('_')[-1].replace('.pth.tar', ''))
                file_list = sorted(file_list, key=sorted_key, reverse=reverse)
                warm_start_file = os.path.join(self.warm_start_file, file_list[0])

            self.detail_logger('Warm start from {}'.format(warm_start_file))

            if not self.disable_cuda and torch.cuda.is_available():
                ws_model = torch.load(warm_start_file)
                self.graph_model.to(self.device)
            else:
                ws_model = torch.load(
                    warm_start_file, map_location=lambda storage, loc: storage)

            # First load model from best, and set it to self._best_model.
            # Use copy to avoid best_model being affected by changes
            self.graph_model.load_state_dict(ws_model['best_state_dict'])
            self._best_model = deepcopy(self.graph_model)
            best_epoch = ws_model['best_epoch']
            if 'best_score' in ws_model.keys():
                best_score = ws_model['best_score']
            else:
                best_score = ws_model['best_mae_error']

            if self.warm_start_best:
                start_epoch = ws_model['best_epoch'] + 1
            else:
                # Warm start from latest model
                self.graph_model.load_state_dict(ws_model['state_dict'])
                start_epoch = ws_model['epoch'] + 1
            self._latest_model = deepcopy(self.graph_model)

            self.optimizer.load_state_dict(ws_model['optimizer'])
            self.target_normalizer.load_state_dict(ws_model['target_normalizer'])
            self.scheduler.load_state_dict(ws_model['scheduler'])

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(best_score)
                print('Now learning rate is: {}.'.format(
                    self.optimizer.param_groups[0]['lr']))
                self.detail_logger('Now learning rate is: {}.'.format(
                    self.optimizer.param_groups[0]['lr']))
                print('Now score is: {}.'.format(best_score))
                self.detail_logger('Now score is: {}.'.format(best_score))
            else:
                self.scheduler.step()

        return start_epoch, best_epoch, best_score

    def _save_model(self, epoch, best_epoch, best_score, optimizer,
                    target_normalizer, scheduler, output_file):
        torch.save({'epoch': epoch + 1,
                    'state_dict': self._latest_model.state_dict(),
                    'best_epoch': best_epoch,
                    'best_state_dict': self._best_model.state_dict(),
                    'best_score': best_score,
                    'optimizer': optimizer.state_dict(),
                    'target_normalizer': target_normalizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                    },
                   output_file.format(best_epoch, best_score, epoch))
