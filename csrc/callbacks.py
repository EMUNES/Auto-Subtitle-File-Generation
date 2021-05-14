# FROM: https://github.com/koukyo1994/kaggle-birdcall-6th-place/blob/master/src/callbacks.py
# THANKS a lot!

"""
Callback functions for catalyst.

For catalyst 21 we need to set:
1. loader_metrics: runner.loader_metrics[self.metric_key]
2. epoch_metricsJ: runner.epoch_metrics[self.loader_key][self.metric_key]
"""

from os import stat
import numpy as np

from typing import List

from catalyst.core import Callback, CallbackOrder, IRunner
from sklearn.metrics import f1_score, average_precision_score, precision_score


class F1Callback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 model_output_key: str = "clipwise_output",
                 prefix: str = "macro_f1"):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.prefix = prefix

    def on_loader_start(self, state: IRunner):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: IRunner):
        targ = state.batch[self.input_key].detach().cpu().numpy()
        out = state.batch[self.output_key]

        clipwise_output = out[self.model_output_key].detach().cpu().numpy()

        self.prediction.append(clipwise_output)
        self.target.append(targ)

        y_pred = clipwise_output.argmax(axis=1)
        y_true = targ.argmax(axis=1)

        score = f1_score(y_true, y_pred, average="macro")
        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: IRunner):
        y_pred = np.concatenate(self.prediction, axis=0).argmax(axis=1)
        y_true = np.concatenate(self.target, axis=0).argmax(axis=1)
        score = f1_score(y_true, y_pred, average="macro")
        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics["valid"] = {
                self.prefix: score
            }
        else:
            state.epoch_metrics["train"] = {
                self.prefix: score
            }            
            
class PrecisionCallback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 model_output_key: str = "clipwise_output",
                 prefix: str = "precision"):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.prefix = prefix

    def on_loader_start(self, state: IRunner):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: IRunner):
        targ = state.batch[self.input_key].detach().cpu().numpy()
        out = state.batch[self.output_key]

        clipwise_output = out[self.model_output_key].detach().cpu().numpy()

        self.prediction.append(clipwise_output)
        self.target.append(targ)

        y_pred = clipwise_output.argmax(axis=1)
        y_true = targ.argmax(axis=1)

        score = precision_score(y_true, y_pred)
        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: IRunner):
        y_pred = np.concatenate(self.prediction, axis=0).argmax(axis=1)
        y_true = np.concatenate(self.target, axis=0).argmax(axis=1)
        score = precision_score(y_true, y_pred)
        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics["valid"] = {
                self.prefix: score
            }
        else:
            state.epoch_metrics["train"] = {
                self.prefix: score
            }  
            
            
class mAPCallback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 model_output_key: str = "clipwise_output",
                 prefix: str = "mAP"):
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.prefix = prefix

    def on_loader_start(self, state: IRunner):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: IRunner):
        targ = state.batch[self.input_key].detach().cpu().numpy()
        out = state.batch[self.output_key]

        clipwise_output = out[self.model_output_key].detach().cpu().numpy()

        self.prediction.append(clipwise_output)
        self.target.append(targ)

        targ = np.nan_to_num(targ)
        clipwise_output = np.nan_to_num(clipwise_output)
        score = average_precision_score(targ, clipwise_output, average=None)
        score = np.nan_to_num(score).mean()
        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: IRunner):
        y_pred = np.concatenate(self.prediction, axis=0)
        y_true = np.concatenate(self.target, axis=0)
        score = average_precision_score(y_true, y_pred, average=None)
        score = np.nan_to_num(score).mean()
        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics["valid"] = {
                self.prefix: score
            }
        else:
            state.epoch_metrics["train"] = {
                self.prefix: score
            }    

