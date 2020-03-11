"""
layout 2.0 (33G raw data, 24G images, 4G masks)
    -hsi_road
         +images (3799 rgb, vis, nir tiff images in uint8, [c, h, w] format)
         +masks (3799 rgb, vis, nir tiff masks in uint8, [h, w] format)
         all.txt (serial number only)
         train.txt (serial number only)
         valid.txt (serial number only)
         vis_correction.txt (already applied)
         nir_correction.txt (already applied)
"""
import os
import numpy as np
import tifffile

import torch
from torch.utils.data import Dataset


class HsiRoadDataset(Dataset):
    CLASSES = ('background', 'road')
    COLLECTION = ('rgb', 'vis', 'nir')

    def __init__(self, data_dir, collection, classes=('background', 'road'), mode='train'):
        # 0 is background and 1 is road
        self.data_dir = data_dir
        self.collection = collection.lower()
        path = os.path.join(data_dir, 'train.txt' if mode == 'train' else 'valid.txt')
        self.name_list = np.genfromtxt(path, dtype='str')
        self.classes = [self.CLASSES.index(cls.lower()) for cls in classes]

    def __getitem__(self, i):
        # pick data
        name = '{}_{}.tif'.format(self.name_list[i], self.collection)

        image_path = os.path.join(self.data_dir, 'images', name)
        mask_path = os.path.join(self.data_dir, 'masks', name)
        image = tifffile.imread(image_path).astype(np.float32) / 255
        mask = tifffile.imread(mask_path).astype(np.long)
        # convert to tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        return image, mask

    def __len__(self):
        return len(self.name_list)


def get_dataset(name, train_mode, kargs):
    ds = None
    if name == 'hsi_road':
        kargs.update({'mode': train_mode})
        ds = HsiRoadDataset(**kargs)
    return ds


