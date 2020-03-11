import torch

from argparse import ArgumentParser
import numpy as np
import os
import yaml

from framework import LightningSeg
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from thop import profile

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

f = open(option.config)
config = yaml.safe_load(f)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------

model = LightningSeg(model_params=config['model_cfgs'], dataset_params=config['dataset_cfgs'],
                    loss_params=config['loss_cfgs'], train_params=config['train_cfgs'])
type_size = 4
params = list(model.parameters())
k = 0
for i in params:
    l = 1
    print("structure:" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("parameters of this layer:" + str(l))
    k = k + l
print("sum of parameters:" + str(k))

print('Model {} : params: {:4f}M'.format(model._get_name(), k * type_size / 1000 / 1000))