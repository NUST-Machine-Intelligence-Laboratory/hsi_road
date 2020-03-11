import os
import shutil
import argparse

import numpy as np
import albumentations as albu
import cv2
import tifffile
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor


class HsiRoadBuilderDataset(Dataset):
    CLASSES = ('background', 'road')
    COLLECTION = ('rgb', 'vis', 'nir')

    def __init__(self, data_dir, classes=('background', 'road')):
        # all list contains all items:
        # (rgb_image, rgb_mask, rgb_visual, vis_image, vis_mask, vis_visual, nir_image, nir_mask, nir_visual)
        f_list = np.genfromtxt(os.path.join(data_dir, 'all_list.txt'), dtype='str')
        # generate lists
        self.rgb_image_list = [os.path.join(data_dir, file) for file in f_list[:, 0]]
        self.rgb_label_list = [os.path.join(data_dir, file) for file in f_list[:, 1]]
        self.vis_image_list = [os.path.join(data_dir, file) for file in f_list[:, 3]]
        self.vis_label_list = [os.path.join(data_dir, file) for file in f_list[:, 4]]
        self.nir_image_list = [os.path.join(data_dir, file) for file in f_list[:, 6]]
        self.nir_label_list = [os.path.join(data_dir, file) for file in f_list[:, 7]]
        # crop to right size, applied to rgb image only
        self.crop_rgb = albu.CenterCrop(704, 1280)   # crop from 736*1280
        self.crop_vis = albu.CenterCrop(256, 480)    # crop from 272*512
        self.crop_nir = albu.CenterCrop(192, 384)    # crop from 217*409
        # parse the mask values
        self.classes = [self.CLASSES.index(cls.lower()) for cls in classes]
        # load correction matrix for hsi data correction
        self.vis_matrix = np.transpose(np.loadtxt(os.path.join(data_dir, 'vis_correction.txt'), dtype=np.float32))
        self.nir_matrix = np.transpose(np.loadtxt(os.path.join(data_dir, 'nir_correction.txt'), dtype=np.float32))

    def __getitem__(self, idx):
        # get file paths
        rgb_image_file, rgb_label_file = self.rgb_image_list[idx], self.rgb_label_list[idx]
        vis_image_file, vis_label_file = self.vis_image_list[idx], self.vis_label_list[idx]
        nir_image_file, nir_label_file = self.nir_image_list[idx], self.nir_label_list[idx]

        # process RGB image and mask
        rgb_img = cv2.cvtColor(cv2.imread(rgb_image_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        rgb_msk = cv2.cvtColor(cv2.imread(rgb_label_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        rgb_msk = self.hsi_mask_parser(rgb_msk)
        rgb_smp = self.crop_rgb(image=rgb_img, mask=rgb_msk)
        rgb_img, rgb_msk = rgb_smp['image'], rgb_smp['mask']
        # transpose to [C H W] formate
        rgb_img = np.transpose(rgb_img, (2, 0, 1))

        # process VIS image and mask
        vis_img = self.demosaic([4, 4], cv2.imread(vis_image_file, cv2.IMREAD_UNCHANGED))
        vis_msk = cv2.cvtColor(cv2.imread(vis_label_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        vis_msk = self.hsi_mask_parser(vis_msk)
        vis_msk = cv2.resize(vis_msk, (507, 272), cv2.INTER_NEAREST)
        # transpose to [C H W] formate
        vis_smp = self.crop_vis(image=vis_img, mask=vis_msk)
        vis_img, vis_msk = vis_smp['image'], vis_smp['mask']
        vis_img = np.transpose(vis_img, (2, 0, 1))

        # process NIR image and mask
        nir_img = self.demosaic([5, 5], cv2.imread(nir_image_file, cv2.IMREAD_UNCHANGED))
        nir_msk = cv2.cvtColor(cv2.imread(nir_label_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        nir_msk = self.hsi_mask_parser(nir_msk)
        nir_msk = cv2.resize(nir_msk, (405, 217), cv2.INTER_NEAREST)
        # transpose to [C H W] formate
        nir_smp = self.crop_nir(image=nir_img, mask=nir_msk)
        nir_img, nir_msk = nir_smp['image'], nir_smp['mask']
        nir_img = np.transpose(nir_img, (2, 0, 1))

        return rgb_img, rgb_msk, vis_img, vis_msk, nir_img, nir_msk

    def __len__(self):
        return len(self.rgb_image_list)

    @staticmethod
    def hsi_mask_parser(mask):
        # make a categorical index mask
        mask = mask[:, :, 0] // 128
        return mask.astype(np.uint8)

    def demosaic(self, cell_size, raw_image):
        # convert mosaic cells into channel images
        # create placeholder for decoded image (channel, hight, width)
        cell_w, cell_h = cell_size[0], cell_size[1]
        raw_h = (raw_image.shape[0] // cell_h) * cell_h
        raw_w = (raw_image.shape[1] // cell_w) * cell_w
        decode_im = np.zeros([cell_w * cell_h, (raw_h // cell_h), (raw_w // cell_w)], dtype=np.float32)
        # fill the placeholder
        for h in range(cell_h):
            for w in range(cell_w):
                decode_im[h * cell_h + w] = raw_image[h:raw_h:cell_h, w:raw_w:cell_w].astype(np.float32)

        # 2048 * 1088
        decode_im = np.transpose(decode_im, [1, 2, 0])
        if cell_w == 4:
            decode_im = decode_im[:, 0:-5, :]
            decode_im = np.matmul(decode_im, self.vis_matrix)
        elif cell_w == 5:
            decode_im = decode_im[:, 4:, :]
            decode_im = np.matmul(decode_im, self.nir_matrix)

        decode_im -= np.min(decode_im)
        decode_im = (decode_im / (np.max(decode_im) + 1e-7)) * 255

        return decode_im.astype(np.uint8)


def run_hsi_builder(src):
    # dataset
    dataset = HsiRoadBuilderDataset(src)
    executor = ThreadPoolExecutor(max_workers=80)
    idxs = [i for i in range(len(dataset))]

    if os.path.isdir('./masks'):
        shutil.rmtree('./masks')

    if os.path.isdir('./images'):
        shutil.rmtree('./images')

    os.mkdir('./masks')
    os.mkdir('./images')

    def save_images(idx):
        rgb_image, rgb_masks, vis_image, vis_masks, nir_image, nir_masks = dataset[idx]
        tifffile.imwrite('images/%06d_rgb.tif' % idx, rgb_image)
        tifffile.imwrite('masks/%06d_rgb.tif' % idx, rgb_masks)
        tifffile.imwrite('images/%06d_vis.tif' % idx, vis_image)
        tifffile.imwrite('masks/%06d_vis.tif' % idx, vis_masks)
        tifffile.imwrite('images/%06d_nir.tif' % idx, nir_image)
        tifffile.imwrite('masks/%06d_nir.tif' % idx, nir_masks)

        return '%06d' % idx

    for ret in executor.map(save_images, idxs):
        print("in main: task {} has finished!".format(ret))


if __name__ == '__main__':
    '''
    e.g. python hsi_builder.py /farm/lhf/data/hsi_road
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='the raw data path')
    args = parser.parse_args()
    run_hsi_builder(args.root)
