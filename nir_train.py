import torch

from argparse import ArgumentParser
import numpy as np
import os
import yaml

from framework import LightningSeg
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def main(options):
    # ------------------------
    # 1 PRE PROCESS THE
    # ------------------------
    f = open(options.config)
    config = yaml.safe_load(f)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------

    model = LightningSeg(model_params=config['model_cfgs'], dataset_params=config['dataset_cfgs'],
                         loss_params=config['loss_cfgs'], train_params=config['train_cfgs'])
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    ckpt_cb = ModelCheckpoint(
        filepath=options.ckpt_path,
        save_best_only=True, verbose=1,
        monitor='v_iou', mode='max', prefix='nir'
    )

    trainer = Trainer(
        default_save_path=options.ckpt_path,
        checkpoint_callback=ckpt_cb,
        early_stop_callback=None,
        gpus=config['train_cfgs']['gpus'],
        distributed_backend=config['train_cfgs']['distribute_backend'],
        nb_gpu_nodes=config['train_cfgs']['nb_gpu_nodes'],
        max_nb_epochs=options.max_nb_epochs,
        nb_sanity_val_steps=0,
        #test_percent_check=0.0,

    )
    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.test(model)


if __name__ == '__main__':
    SEED = 2334
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser()

    # checkpointing and logging
    parser.add_argument('--ckpt_path', type=str, default=os.getcwd())
    parser.add_argument('--load_pretrain', type=bool, default=False)

    # model and training setup
    parser.add_argument('--config', type=str, help='name a yaml running configuration')
    parser.add_argument('--max_nb_epochs', type=int, default=100)
    option = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(option)
