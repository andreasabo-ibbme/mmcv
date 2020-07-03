# Copyright (c) Open-MMLab. All rights reserved.
import logging
import os.path as osp
import time

import torch
import csv

import mmcv
from .checkpoint import load_checkpoint, save_checkpoint
from .dist_utils import get_dist_info
from .hooks import HOOKS, Hook, IterTimerHook
from .log_buffer import LogBuffer
from .priority import get_priority
from .utils import get_host_info, get_time_str, obj_from_dict
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from .pytorchtools import EarlyStopping
import os
# from ...early_stopping_pytorch/pytorchtools import EarlyStopping

def weight_reset(m):
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()


class Runner(object):
    """A training helper for PyTorch.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
    """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None,
                 meta=None, 
                 things_to_log=None,
                 early_stopping=False,
                 force_run_all_epochs=True, 
                 es_patience=10, 
                 es_start_up=50):
        assert callable(batch_processor)
        self.model = model
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        self.batch_processor = batch_processor
        self.things_to_log = things_to_log
        self.early_stopping = early_stopping
        self.force_run_all_epochs = force_run_all_epochs


        self.es_patience = es_patience
        self.es_start_up = es_start_up

        # create work_dir
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()



        if logger is None:
            print('Andrea - making new logger')
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        if meta is not None:
            assert isinstance(meta, dict), '"meta" must be a dict or None'
        self.meta = meta

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(optimizer, torch.optim,
                                      dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                f'but got {type(optimizer)}')
        return optimizer

    def _add_file_handler(self,
                          logger,
                          filename=None,
                          mode='w',
                          level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, log_dir=None, level=logging.INFO):
        """Init the logger.

        Args:
            log_dir(str, optional): Log file directory. If not specified, no
                log file will be used.
            level (int or str): See the built-in python logging module.

        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        print("logger in here: ", __name__)
        if log_dir and self.rank == 0:
            filename = f'{self.timestamp}.log'
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    def current_momentum(self):
        """Get current momentums.

        Returns:
            list: Current momentum of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        momentums = []
        for group in self.optimizer.param_groups:
            if 'momentum' in group.keys():
                momentums.append(group['momentum'])
            elif 'betas' in group.keys():
                momentums.append(group['betas'][0])
            else:
                momentums.append(0)
        return momentums

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        true_labels, predicted_labels, pred_raw = [], [], []
        batch_loss = 0

        self.call_hook('before_train_epoch')

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            outputs, raw, overall_loss = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)

            # If we get a nan in the loss, just ignore it
            if not np.isnan(overall_loss):
                batch_loss += overall_loss*len(raw['true'])
            else:
                print('got non in training loss')
            # print(true_labels, "vs. ", raw['true'])
            true_labels.extend(raw['true'])
            predicted_labels.extend(raw['pred'])
            pred_raw.extend(raw['raw_preds'])
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])

                                  
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1
        self._epoch += 1
        print("end model train epoch")

        # true_labels, predicted_labels = self.remove_non_labelled_data(true_labels, predicted_labels)
        # print(len(true_labels), true_labels)
        # print(len(predicted_labels), predicted_labels)
        acc = accuracy_score(true_labels, predicted_labels)
        log_this = {'accuracy': acc}
        self.log_buffer.update(log_this, 1) 

        self.preds = predicted_labels
        self.labels = true_labels
        self.preds_raw = pred_raw
        self.call_hook('after_val_epoch')

        self.call_hook('after_train_epoch')
        # print('train', 'labels', true_labels, 'oreds', predicted_labels)

        # print(len(true_labels), len(predicted_labels))
        # print(true_labels)
        # print(predicted_labels)
        # print("what is this (train): ", accuracy_score(true_labels, predicted_labels))
        print("end training epoch")
        return true_labels, predicted_labels


    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        true_labels, predicted_labels, pred_raw = [], [], []
        batch_loss = 0

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs, raw, overall_loss = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs)
                true_labels.extend(raw['true'])
                predicted_labels.extend(raw['pred'])
                pred_raw.extend(raw['raw_preds'])
                if not np.isnan(overall_loss):
                    batch_loss += overall_loss*len(raw['true'])

            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')
        

        batch_loss = batch_loss / len(true_labels)
        # true_labels, predicted_labels = self.remove_non_labelled_data(true_labels, predicted_labels)
        acc = accuracy_score(true_labels, predicted_labels)
        log_this = {'accuracy': acc}
        self.log_buffer.update(log_this, 1) 

        self.preds = predicted_labels
        self.labels = true_labels
        self.preds_raw = pred_raw
        # print('labels', true_labels, 'preds', predicted_labels)

        if self.early_stopping and not self.early_stopping_obj.early_stop and self.epoch >= self.es_start_up:
            self.es_before_step = self.early_stopping_obj.early_stop
            self.early_stopping_obj(batch_loss, self.model)

            if self.es_before_step == False and self.early_stopping_obj.early_stop == True:
                self.early_stopping_epoch = self.epoch - self.es_patience

                self.log_buffer.update({'stop_epoch_val': self.early_stopping_epoch}, 1)
                print("Updated the buffer with the stop epoch: ", self.early_stopping_epoch)
        
        if not self.early_stopping and self.epoch == self._max_epochs: #dont have early stopping
            torch.save(self.model.state_dict(), self.es_checkpoint)


        self.call_hook('after_val_epoch')

        return true_labels, predicted_labels


    def test(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'test'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        true_labels, predicted_labels, pred_raw = [], [], []
        batch_loss = 0

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs, raw, overall_loss = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs)
                true_labels.extend(raw['true'])
                predicted_labels.extend(raw['pred'])
                pred_raw.extend(raw['raw_preds'])
                if not np.isnan(overall_loss):
                    batch_loss += overall_loss*len(raw['true'])

            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')
        


        # true_labels, predicted_labels = self.remove_non_labelled_data(true_labels, predicted_labels)
        acc = accuracy_score(true_labels, predicted_labels)
        log_this = {'accuracy': acc}
        self.log_buffer.update(log_this, 1) 

        self.preds = predicted_labels
        self.labels = true_labels
        self.preds_raw = pred_raw
        self.call_hook('after_val_epoch')
        # print('test', 'labels', true_labels, 'oreds', predicted_labels)


        return true_labels, predicted_labels


    def remove_non_labelled_data(self, true_labels, pred_labels):
        true_np = np.asarray(true_labels)
        pred_np = np.asarray(pred_labels)
        keep = np.argwhere(true_np >= 0 ).transpose().squeeze()
        # keep = np.transpose(keep).squeeze()
        pred_labels = pred_np[keep]
        true_labels = true_np[keep]
        return list(true_labels), list(pred_labels)


    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def basic_no_log_eval(self, data_loader, **kwargs):
        self.model.eval()
        self.data_loader = data_loader
        true_labels, predicted_labels, pred_raw = [], [], []
        batch_loss = 0

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            with torch.no_grad():
                outputs, raw, overall_loss = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs)
                true_labels.extend(raw['true'])
                predicted_labels.extend(raw['pred'])
                pred_raw.extend(raw['raw_preds'])
                batch_loss += overall_loss*len(raw['true'])


        return true_labels, predicted_labels, pred_raw

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        not_done = True
        while not_done:
            print('===================starting training...=========================')
            print(kwargs)
            # Reset the epoch counters
            self.mode = None
            self._epoch = 0
            self._iter = 0
            self._inner_iter = 0
            self._max_epochs = 0
            self._max_iters = 0

            try: 
                print("Starting training for ", self.work_dir)
                assert isinstance(data_loaders, list)
                assert mmcv.is_list_of(workflow, tuple)
                assert len(data_loaders) == len(workflow)

                es_checkpoint = self.work_dir + '/checkpoint.pt'
                self.es_checkpoint = es_checkpoint
                if self.early_stopping:
                    self.early_stopping_obj = EarlyStopping(patience=self.es_patience, verbose=True, path=es_checkpoint)

                self._max_epochs = max_epochs
                for i, flow in enumerate(workflow):
                    mode, epochs = flow
                    if mode == 'train':
                        self._max_iters = self._max_epochs * len(data_loaders[i])
                        break

                work_dir = self.work_dir if self.work_dir is not None else 'NONE'
                self.logger.info('Start running, host: %s, work_dir: %s',
                                get_host_info(), work_dir)
                self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
                self.call_hook('before_run')
                print("work dir: ", self.work_dir)
                train_accs = np.zeros((1, max_epochs)) * np.nan
                val_accs = np.zeros((1, max_epochs)) * np.nan
                
                columns = ['epoch', 'train_acc', 'val_acc']
                df_all = pd.DataFrame(columns=columns)

                while self.epoch < max_epochs:
                    for i, flow in enumerate(workflow):
                        mode, epochs = flow
                        if isinstance(mode, str):  # self.train()
                            if not hasattr(self, mode):
                                raise ValueError(
                                    f'runner has no method named "{mode}" to run an '
                                    'epoch')
                            epoch_runner = getattr(self, mode)
                        elif callable(mode):  # custom train()
                            epoch_runner = mode
                        else:
                            raise TypeError('mode in workflow must be a str or '
                                            f'callable function, not {type(mode)}')
                        for _ in range(epochs):
                            if mode == 'train' and self.epoch >= max_epochs:
                                return
                            true_labels, predicted_labels = epoch_runner(data_loaders[i], **kwargs)

                            acc = accuracy_score(true_labels, predicted_labels)

                            if mode == 'train':
                                df_all.loc[len(df_all)] = [self.epoch-1, acc, val_accs[0, self.epoch - 1]]

                            elif mode == 'val':
                                val_accs[0, self.epoch-1] = acc
                                df_all.loc[df_all['epoch'] == self.epoch-1,'val_acc'] = acc


                    if self.early_stopping:
                        if not self.force_run_all_epochs and self.early_stopping_obj.early_stop:
                            break
                    df_all.to_csv(self.work_dir + "/results_df.csv")

                    # We have successfully finished this participant
                    not_done = False
            except Exception as e: 
                not_done = True
                logging.exception("loss calc message=================================================")

                # Reset the model parameters
                print("======================================going to retrain again, resetting parameters...")
                print("This is the error we got:", e)
                try:
                    self.model.module.apply(weight_reset)
                    print('successfully reset weights')
                except Exception as e: 
                    print("This is the error we got _ 2:", e)


        # If we stopped early, evaluate the performance of the saved model on all datasets
        if self.early_stopping:
            self.log_buffer.update({'early_stop_epoch': self.early_stopping_epoch}, 1) 

            print('stopped at epoch: ', self.early_stopping_epoch)

            print("*****************************now doing eval: ")
            print("workflow", workflow)
            print("data_loaders", data_loaders)
            self.early_stop_eval(es_checkpoint, workflow, data_loaders, **kwargs)



        time.sleep(10)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')





    def early_stop_eval(self, es_checkpoint, workflow, data_loaders, **kwargs):
        self.model.load_state_dict(torch.load(es_checkpoint))
        self.model.eval()

        for i, flow in enumerate(workflow):
            mode, _ = flow

            # mode = "train", "val", "test"
            true_labels, predicted_labels, raw_preds = self.basic_no_log_eval(data_loaders[i], **kwargs)
            acc = accuracy_score(true_labels, predicted_labels)


            final_results_base, amb = os.path.split(self.work_dir)
            final_results_path = os.path.join(final_results_base, 'all_eval', self.things_to_log['wandb_group'])
            if mode == 'test':
                final_results_file = os.path.join(final_results_path,'test.csv')
            if mode == 'val':
                final_results_file = os.path.join(final_results_path,'val.csv')
            if mode == 'train':
                final_results_file = os.path.join(final_results_path,'train.csv')           

            print("saving to ", final_results_file)
            mmcv.mkdir_or_exist(final_results_path)
            header = ['amb', 'true_score', 'pred_round', 'pred_raw']

            if not os.path.exists(final_results_file):
                with open (final_results_file,'w') as f:                            
                    writer = csv.writer(f, delimiter=',') 
                    writer.writerow(header)


            with open (final_results_file,'a') as f:                            
                writer = csv.writer(f, delimiter=',') 
                for num in range(len(true_labels)):
                    writer.writerow([amb, true_labels[num], predicted_labels[num], raw_preds[num]])



    def register_lr_hook(self, lr_config):
        if isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater updater.
            # Since this is not applicable for `CosineAnealingLrUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = mmcv.build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook)

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook)

    def register_checkpoint_hook(self, checkpoint_config):
        if checkpoint_config is None:
            return
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        self.register_hook(hook)

    def register_momentum_hook(self, momentum_config):
        if momentum_config is None:
            return
        if isinstance(momentum_config, dict):
            assert 'policy' in momentum_config
            policy_type = momentum_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of momentum updater.
            # Since this is not applicable for `CosineAnealingMomentumUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'MomentumUpdaterHook'
            momentum_config['type'] = hook_type
            hook = mmcv.build_from_cfg(momentum_config, HOOKS)
        else:
            hook = momentum_config
        self.register_hook(hook)

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = mmcv.build_from_cfg(
                info, HOOKS, default_args=dict(interval=log_interval, initial_config=self.things_to_log))
            self.register_hook(logger_hook, priority='VERY_LOW')

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - MomentumUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        self.register_lr_hook(lr_config)
        self.register_momentum_hook(momentum_config)
        self.register_optimizer_hook(optimizer_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_hook(IterTimerHook())
        self.register_logger_hooks(log_config)
