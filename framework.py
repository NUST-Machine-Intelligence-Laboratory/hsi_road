import pytorch_lightning as pl
from collections import OrderedDict

import numpy as np
import imageio

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from datasets import get_dataset
from segmentations import get_model
from losses import get_loss
from optimizer import Ranger

from metrics import IoUMetric


class LightningSeg(pl.LightningModule):

    def __init__(self, model_params, dataset_params, loss_params, train_params):

        super(LightningSeg, self).__init__()

        self.model_cfgs = model_params
        self.dataset_cfgs = dataset_params
        self.loss_cfgs = loss_params
        self.train_cfgs = train_params

        self.model = get_model(self.model_cfgs['model_name'], self.model_cfgs['model_args'])
        self.loss = get_loss(self.loss_cfgs['loss_name'], self.loss_cfgs['loss_args'])
        self.metric = IoUMetric(activation=self.loss_cfgs['loss_activation'])

        self.use_parallel = (self.train_cfgs['distribute_backend'] == 'dp' or self.train_cfgs['distribute_backend'] == 'ddp2')
        self.use_sampler = (self.train_cfgs['distribute_backend'] is not None)
        self.nb_gpus = len([int(x.strip()) for x in self.train_cfgs['gpus'].split(',')])

    def forward(self, x):
        return self.model(x)

    def training_step(self, item, idx):
        x, y = item
        y_hat = self.forward(x)
        t_loss = self.loss(y_hat, y)
        t_iou = self.metric(y_hat, y)[1]

        # when using distribution strategy, unpack the loss values
        if self.use_parallel:
            t_loss = t_loss.squeeze(0)
            t_iou = t_iou.squeeze(0)

        output = OrderedDict({
            'loss': t_loss,
            't_iou': t_iou,
            'progress_bar': {'t_iou': t_iou},
            'log': {'t_loss': t_loss, 't_iou': t_iou}
        })
        return output

    def validation_step(self, item, idx):
        x, y = item
        y_hat = self.forward(x)
        v_loss = self.loss(y_hat, y)
        v_iou = self.metric(y_hat, y)[1]

        if self.use_parallel:
            v_loss = v_loss.squeeze(0)
            v_iou = v_iou.squeeze(0)

        output = OrderedDict({
            'v_loss': v_loss,
            'v_iou': v_iou
        })
        return output

    def validation_end(self, outputs):
        show = dict()
        for metric_name in ['v_loss', 'v_iou']:
            metric_sum = 0
            for output in outputs:
                metric_value = output[metric_name]
                if self.use_parallel:
                    metric_value = torch.mean(metric_value)
                metric_sum += metric_value
            show[metric_name] = metric_sum / len(outputs)
        result = {'progress_bar': show, 'log': show, 'v_loss': show['v_loss']}
        return result

    def test_step(self, item, idx):
        x, y = item
        y_hat = self.forward(x)
        v_iou = self.metric(y_hat, y)[1]

        y = torch.squeeze(y, dim=0).cpu().numpy()[1, ...].astype(np.uint8) * 128
        y_hat = torch.squeeze(torch.argmax(y_hat, dim=1), dim=0).cpu().numpy().astype(np.uint8) * 128
        z = np.zeros_like(y, dtype=y.dtype)
        im = np.stack([y, y_hat, z], axis=-1)

        imageio.imwrite(('test/%d.png' % idx), im)

        output = OrderedDict({'id': idx, 'iou': v_iou})
        return output

    def test_end(self, outputs):
        f = open('test/lists.txt', 'w')
        for output in outputs:
            line = '{:d},{:f}\r\n'.format(output['id'], output['iou'])
            f.write(line)
        return

    def configure_optimizers(self):
        learning_rate = self.train_cfgs['lr_rate']
        scheduler_step = self.train_cfgs['lr_scheduler_step']
        scheduler_gamma = self.train_cfgs['lr_scheduler_gamma']
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        ds = get_dataset(self.dataset_cfgs['dataset_name'], 'train', self.dataset_cfgs['dataset_args'])
        sampler = DistributedSampler(ds) if self.use_sampler else None
        loader = DataLoader(
            dataset=ds,
            batch_size=self.train_cfgs['batch_size_per_gpu'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.nb_gpus*self.train_cfgs['batch_size_per_gpu']*2
        )
        return loader

    @pl.data_loader
    def val_dataloader(self):
        ds = get_dataset(self.dataset_cfgs['dataset_name'], 'valid', self.dataset_cfgs['dataset_args'])
        loader = DataLoader(
            dataset=ds,
            batch_size=self.train_cfgs['batch_size_per_gpu'],
            shuffle=False,
            num_workers=self.nb_gpus*self.train_cfgs['batch_size_per_gpu']
        )
        return loader

    @pl.data_loader
    def test_dataloader(self):
        ds = get_dataset(self.dataset_cfgs['dataset_name'], 'valid', self.dataset_cfgs['dataset_args'])
        loader = DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=1)
        return loader

    def tng_dataloader(self):
        pass









