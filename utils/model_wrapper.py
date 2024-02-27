from abc import ABC
from typing import Optional
import numpy as np
import torch
import logging
from tqdm import tqdm
from torcheval import metrics as M


class ModelWrapper:

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.schedule_monitor = None
        self.schedule = None
        self.mloss = None
        self.model = model
        self.optimizer = None
        self.loss_fcn = None
        self.metrics = None
        self.device = device

    def to(self,  *args, **kwargs):
        self.model.to( *args, **kwargs)

    def compile(self, optimizer=None, loss_fcn=None, metrics=None, schedule=None, schedule_monitor=None):
        self.optimizer = optimizer
        self.loss_fcn = loss_fcn
        self.metrics = metrics
        self.mloss = M.Mean()
        self.schedule = schedule
        self.schedule_monitor = schedule_monitor
        self.model.to(self.device)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def run_step(self, data):
        if len(data) == 2:
            inputs, labels = data
            sample_weights = None
        else:
            inputs, labels, sample_weights = data

        inputs = inputs.to(self.device)
        preds = self.model(inputs)
        if sample_weights is None:
            loss = self.loss_fcn(preds.to('cpu'), labels)
        else:
            loss = self.loss_fcn(preds.to('cpu'), labels, sample_weights)

        self.mloss.update(loss)
        preds = preds.squeeze()
        labels = labels.squeeze()

        for metric in self.metrics:
            metric.update(preds, labels)
        return loss

    def train_step(self, data):
        loss = self.run_step(data)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def test_step(self, data):
        with torch.no_grad():
            loss = self.run_step(data)
        return loss

    def run_epoch(self, data_loader, training=False, verbose=1):
        size = len(data_loader)
        self.mloss.reset()
        for metric in self.metrics:
            metric.reset()

        if verbose > 0:
            bar = tqdm(total=size)
        else:
            bar = None

        for data in data_loader:
            if training:
                self.train_step(data)
            else:
                self.test_step(data)

            if verbose > 0:
                mloss = self.mloss.compute().item()
                disp = 'loss: {:.4f} '.format(mloss)
                for metric in self.metrics:
                    m = metric.compute().item()
                    if hasattr(metric, 'name'):
                        mn = ' ' + metric.name + ':'
                    else:
                        mn = ''
                    disp += (mn + ' {:.4f} '.format(m))
                bar.set_description(disp)
                bar.update()
        if verbose > 0:
            bar.close()

        record = [self.mloss.compute().item()]
        for metric in self.metrics:
            record.append(metric.compute().item())

        return record

    def train_epoch(self, data_loader, verbose=1):
        return self.run_epoch(data_loader, True, verbose)

    def test_epoch(self, data_loader, verbose=1):
        return self.run_epoch(data_loader, False, verbose)

    def evaluate(self, test_data, verbose=1):
        rtv = self.test_epoch(test_data, verbose)
        return rtv

    def predict(self, data_loader, verbose=1):
        size = len(data_loader)

        if verbose > 0:
            bar = tqdm(total=size)
        else:
            bar = None

        results = []
        with torch.no_grad():
            for data in data_loader:
                inputs = data.to(self.device)
                preds = self.model(inputs)
                results.append(preds.to('cpu').numpy())
                if verbose > 0:
                    bar.update()

        if verbose > 0:
            bar.close()

        return np.concatenate(results, axis=0).squeeze()


    def train(self, train_data, val_data=None, epochs=1, save_path=None, verbose=1):
        hist = {'train': [], 'val': []}
        val_loss = 0

        for epoch in range(epochs):

            if verbose > 0:
                print('Epoch {:d}/{:d} start training.'.format(epoch + 1, epochs))
            rtv = self.train_epoch(train_data, verbose)
            hist['train'].append(rtv)

            if val_data is not None:
                if verbose > 0:
                    print('Epoch {:d}/{:d} start testing.'.format(epoch + 1, epochs))
                rtv = self.test_epoch(val_data, verbose)
                hist['val'].append(rtv)
                val_loss = rtv[0]

            if self.schedule is not None:
                if self.schedule_monitor == 'val_loss':
                    self.schedule.step(val_loss)
                else:
                    self.schedule.step()

            if save_path is not None:
                torch.save(self.model.state_dict(), save_path)
                if verbose > 0:
                    print('Save model to {:s}'.format(save_path))
            if verbose > 0:
                print()

        hist['train'] = np.array(hist['train'])
        hist['val'] = np.array(hist['val'])
        return hist


class BinaryAUC(M.BinaryAUROC):
    def __init__(self, name='auc', *args, **kwargs):
        self.name = name
        super(BinaryAUC, self).__init__(*args, **kwargs)


class BinaryAccuracy(M.BinaryAccuracy):
    def __init__(self, name='acc', *args, **kwargs):
        self.name = name
        super(BinaryAccuracy, self).__init__(*args, **kwargs)


class __BinaryConfusionMatrix(M.Metric, ABC):
    def __init__(
            self, name='acc',
            threshold: float = 0.5,
            device: Optional[torch.device] = None,
    ):
        super().__init__(device=device)
        self.name = name
        self.threshold = threshold

        self._add_state('num_tp', torch.tensor(0.0, device=self.device))
        self._add_state('num_fn', torch.tensor(0.0, device=self.device))
        self._add_state('num_fp', torch.tensor(0.0, device=self.device))
        self._add_state('num_tn', torch.tensor(0.0, device=self.device))

    @torch.inference_mode()
    def update(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        inputs = torch.where(inputs < self.threshold, 0, 1)
        targets = torch.where(targets < self.threshold, 0, 1)
        ((tn, fp), (fn, tp)) = M.functional.binary_confusion_matrix(inputs, targets, threshold=self.threshold)
        self.num_tp += tp
        self.num_fn += fn
        self.num_fp += fp
        self.num_tn += tn
        return self

    @torch.inference_mode()
    def merge_state(self, metrics):
        for metric in metrics:
            self.num_tp += metric.num_tp.to(self.device)
            self.num_fn += metric.num_fn.to(self.device)
            self.num_fp += metric.num_fp.to(self.device)
            self.num_tn += metric.num_tn.to(self.device)
        return self


class APCER(__BinaryConfusionMatrix):
    def __init__(self, name='apcer', threshold: float = 0.5, device: Optional[torch.device] = None):
        super().__init__(name, threshold, device)

    @torch.inference_mode()
    def compute(self):
        val = self.num_fp / (self.num_tn + self.num_fp)
        if torch.isnan(val):
            logging.warning(
                "FAR is converted from NaN to 0s."
            )
            return torch.nan_to_num(val)
        return val


class BPCER(__BinaryConfusionMatrix):
    def __init__(self, name='bpcer', threshold: float = 0.5, device: Optional[torch.device] = None):
        super().__init__(name, threshold, device)

    @torch.inference_mode()
    def compute(self):
        val = self.num_fn / (self.num_tp + self.num_fn)
        if torch.isnan(val):
            logging.warning(
                "FAR is converted from NaN to 0s."
            )
            return torch.nan_to_num(val)
        return val