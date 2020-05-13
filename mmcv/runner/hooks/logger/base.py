# Copyright (c) Open-MMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from ..hook import Hook


import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.multiclass import unique_labels




class LoggerHook(Hook):
    """Base class for logger hooks.

    Args:
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging.
    """

    __metaclass__ = ABCMeta

    def __init__(self, interval=10, ignore_last=True, reset_flag=False, initial_config=None):
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag
        self.initial_config = initial_config
    @abstractmethod
    def log(self, runner):
        pass

    def before_run(self, runner):
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def before_epoch(self, runner):
        runner.log_buffer.clear()  # clear logs of last epoch

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_train_epoch(self, runner):
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_val_epoch(self, runner):
        if runner._epoch % 5 == 0:
            # class_names = np.array([str(x) for x in range(10)])
            num_class = runner.things_to_log['num_class']
            class_names = [str(i) for i in range(num_class)]
            fig_title = runner.mode.upper() + " Confusion matrix, epoch: " + str(runner._epoch)
            fig = self.plot_confusion_matrix(runner.labels, runner.preds, class_names, False, fig_title)


            figure_name = runner.work_dir +"/" + runner.mode + "_" + str(runner._epoch)+  ".png"

            fig.savefig(figure_name)

            runner.log_buffer.logChart(fig, runner.mode + "_" + str(runner._epoch)+  ".png")
        # print("predictions: ", runner.preds)
        runner.log_buffer.average()
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()
            

    def plot_confusion_matrix(self, y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape[1] is not len(classes):
            # print("our CM is not the right size!!")
            all_labels = y_true + y_pred
            y_all_unique = list(set(all_labels))
            y_all_unique.sort()

            cm_new = np.zeros((len(classes), len(classes)), dtype=np.int64)
            for i in range(len(y_all_unique)):
                for j in range(len(y_all_unique)):
                    i_global = y_all_unique[i]
                    j_global = y_all_unique[j]
                    cm_new[i_global, j_global] = cm[i,j]
                   

            cm = cm_new


        # print(cm)
        # classes = classes[unique_labels(y_true, y_pred).astype(int)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # print("Normalized confusion matrix")
        # else:
            # print('Confusion matrix, without normalization')
# 
        #print(cm)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange( cm.shape[1]),
            yticks=np.arange( cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')
        
        ax.set_xlim(-0.5, cm.shape[1]-0.5)
        ax.set_ylim(cm.shape[0]-0.5, -0.5)

        # Rotate the tick labels and set their alignment.
        # print(ax.get_xticklabels())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return fig

        