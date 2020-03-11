import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import numpy as np


class CatBnAct(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(self, num_init_features, kernel_size=7,
                 padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(
            3, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(
            self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(nn.Module):
    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
                 b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                 num_classes=1000, test_time_pool=False):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4

        blocks = OrderedDict()

        # conv1
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=3, padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7, padding=3)

        # conv2
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs)

        self.features = nn.Sequential(blocks)

        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        self.last_linear = nn.Conv2d(in_chs, num_classes, kernel_size=1, bias=True)

    def logits(self, features):
        if not self.training and self.test_time_pool:
            x = F.avg_pool2d(features, kernel_size=7, stride=1)
            out = self.last_linear(x)
            # The extra test time pool should be pooling an img_size//32 - 6 size patch
            out = adaptive_avgmax_pool2d(out, pool_type='avgmax')
        else:
            x = adaptive_avgmax_pool2d(features, pool_type='avg')
            out = self.last_linear(x)
        return out.view(out.size(0), -1)

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

class DPNEncorder(DPN):

    def __init__(self, feature_blocks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_blocks = np.cumsum(feature_blocks)
        self.pretrained = False

        del self.last_linear

    def forward(self, x):

        features = []

        input_block = self.features[0]

        x = input_block.conv(x)
        x = input_block.bn(x)
        x = input_block.act(x)
        features.append(x)

        x = input_block.pool(x)

        for i, module in enumerate(self.features[1:], 1):
            x = module(x)
            if i in self.feature_blocks:
                features.append(x)

        out_features = [
            features[4],
            F.relu(torch.cat(features[3], dim=1), inplace=True),
            F.relu(torch.cat(features[2], dim=1), inplace=True),
            F.relu(torch.cat(features[1], dim=1), inplace=True),
            features[0],
        ]

        return out_features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        super().load_state_dict(state_dict, **kwargs)


dpn_encoders = {
    'dpn68': {
        'encoder': DPNEncorder,
        'out_shapes': (832, 704, 320, 144, 10),
        'params': {
            'feature_blocks': (3, 4, 12, 4),
            'groups': 32,
            'inc_sec': (16, 32, 32, 64),
            'k_r': 128,
            'k_sec': (3, 4, 12, 3),
            'num_classes': 1000,
            'num_init_features': 10,
            'small': True,
            'test_time_pool': True
        },
    },

    'dpn68b': {
        'encoder': DPNEncorder,
        'out_shapes': (832, 704, 320, 144, 10),
        'params': {
            'feature_blocks': (3, 4, 12, 4),
            'b': True,
            'groups': 32,
            'inc_sec': (16, 32, 32, 64),
            'k_r': 128,
            'k_sec': (3, 4, 12, 3),
            'num_classes': 1000,
            'num_init_features': 10,
            'small': True,
            'test_time_pool': True,
        },
    },

    'dpn92': {
        'encoder': DPNEncorder,
        'out_shapes': (2688, 1552, 704, 336, 64),
        'params': {
            'feature_blocks': (3, 4, 20, 4),
            'groups': 32,
            'inc_sec': (16, 32, 24, 128),
            'k_r': 96,
            'k_sec': (3, 4, 20, 3),
            'num_classes': 1000,
            'num_init_features': 64,
            'test_time_pool': True
        },
    },

    'dpn98': {
        'encoder': DPNEncorder,
        'out_shapes': (2688, 1728, 768, 336, 96),
        'params': {
            'feature_blocks': (3, 6, 20, 4),
            'groups': 40,
            'inc_sec': (16, 32, 32, 128),
            'k_r': 160,
            'k_sec': (3, 6, 20, 3),
            'num_classes': 1000,
            'num_init_features': 96,
            'test_time_pool': True,
        },
    },

    'dpn107': {
        'encoder': DPNEncorder,
        'out_shapes': (2688, 2432, 1152, 376, 128),
        'params': {
            'feature_blocks': (4, 8, 20, 4),
            'groups': 50,
            'inc_sec': (20, 64, 64, 128),
            'k_r': 200,
            'k_sec': (4, 8, 20, 3),
            'num_classes': 1000,
            'num_init_features': 128,
            'test_time_pool': True
        },
    },

    'dpn131': {
        'encoder': DPNEncorder,
        'out_shapes': (2688, 1984, 832, 352, 128),
        'params': {
            'feature_blocks': (4, 8, 28, 4),
            'groups': 40,
            'inc_sec': (16, 32, 32, 128),
            'k_r': 160,
            'k_sec': (4, 8, 28, 3),
            'num_classes': 1000,
            'num_init_features': 128,
            'test_time_pool': True
        },
    },

}
