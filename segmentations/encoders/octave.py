from functools import partial

import torch.nn as nn
import torch.nn.functional as F

# ToDo this module need more modification to adapt to regular encoder


class OctConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, alpha_in=0.25, alpha_out=0.25, type='normal'):
        super(OctConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        hf_ch_in = int(in_channels * (1 - alpha_in))
        hf_ch_out = int(out_channels * (1 - alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        if type == 'first':
            if stride == 2:
                self.downsample = nn.AvgPool2d(kernel_size=2, stride=stride)
            self.convh = nn.Conv2d(in_channels, hf_ch_out, kernel_size=kernel_size, stride=1, padding=padding)
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.convl = nn.Conv2d(in_channels, lf_ch_out, kernel_size=kernel_size, stride=1, padding=padding)
        elif type == 'last':
            if stride == 2:
                self.downsample = nn.AvgPool2d(kernel_size=2, stride=stride)
            self.convh = nn.Conv2d(hf_ch_in, out_channels, kernel_size=kernel_size, padding=padding)
            self.convl = nn.Conv2d(lf_ch_in, out_channels, kernel_size=kernel_size, padding=padding)
            self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
        else:
            if stride == 2:
                self.downsample = nn.AvgPool2d(kernel_size=2, stride=stride)
            self.L2L = nn.Conv2d(lf_ch_in, lf_ch_out, kernel_size=kernel_size, stride=1, padding=padding)
            self.L2H = nn.Conv2d(lf_ch_in, hf_ch_out, kernel_size=kernel_size, stride=1, padding=padding)
            self.H2L = nn.Conv2d(hf_ch_in, lf_ch_out, kernel_size=kernel_size, stride=1, padding=padding)
            self.H2H = nn.Conv2d(hf_ch_in, hf_ch_out, kernel_size=kernel_size, stride=1, padding=padding)
            self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
            self.avg_pool = partial(F.avg_pool2d, kernel_size=2, stride=2)

    def forward(self, x):
        if self.type == 'first':
            if self.stride == 2:
                x = self.downsample(x)
            hf = self.convh(x)
            lf = self.avg_pool(x)
            lf = self.convl(lf)

            return hf, lf
        elif self.type == 'last':
            hf, lf = x
            if self.stride == 2:
                hf = self.downsample(hf)
                return self.convh(hf) + self.convl(lf)
            else:
                return self.convh(hf) + self.convl(self.upsample(lf))
        else:
            hf, lf = x
            if self.stride == 2:
                hf = self.downsample(hf)
                return self.H2H(hf) + self.L2H(lf), self.L2L(F.avg_pool2d(lf, kernel_size=2, stride=2)) + self.H2L(self.avg_pool(hf))
            else:
                return self.H2H(hf) + self.upsample(self.L2H(lf)), self.L2L(lf) + self.H2L(self.avg_pool(hf))


def norm_conv3x3(in_planes, out_planes, stride=1, type=None):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def norm_conv1x1(in_planes, out_planes, stride=1, type=None):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def oct_conv3x3(in_planes, out_planes, stride=1, type='normal'):
    """3x3 convolution with padding"""
    return OctConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, type=type)


def oct_conv1x1(in_planes, out_planes, stride=1, type='normal'):
    """1x1 convolution"""
    return OctConv(in_planes, out_planes, kernel_size=1, stride=stride, type=type)


class _BatchNorm2d(nn.Module):
    def __init__(self, num_features, alpha_in=0.25, alpha_out=0.25, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm2d, self).__init__()
        hf_ch = int(num_features * (1 - alpha_in))
        lf_ch = num_features - hf_ch
        self.bnh = nn.BatchNorm2d(hf_ch)
        self.bnl = nn.BatchNorm2d(lf_ch)

    def forward(self, x):
        hf, lf = x
        return self.bnh(hf), self.bnl(lf)


class _ReLU(nn.ReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(_ReLU, self).forward(hf)
        lf = super(_ReLU, self).forward(lf)
        return hf, lf


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, type="normal", oct_conv_on=True):
        super(BasicBlock, self).__init__()
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        norm_func = _BatchNorm2d if oct_conv_on else nn.BatchNorm2d
        act_func = _ReLU if oct_conv_on else nn.ReLU

        self.conv1 = conv3x3(inplanes, planes, type="first" if type == "first" else "normal")
        self.bn1 = norm_func(planes)
        self.relu1 = act_func(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride, type="last" if type == "last" else "normal")
        if type == "last":
            norm_func = nn.BatchNorm2d
            act_func = nn.ReLU
        self.bn2 = norm_func(planes)
        self.relu2 = act_func(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if isinstance(out, (tuple, list)):
            assert len(out) == len(identity) and len(out) == 2
            out = (out[0] + identity[0], out[1] + identity[1])
        else:
            out += identity

        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, type="normal", oct_conv_on=True):
        super(Bottleneck, self).__init__()
        conv1x1 = oct_conv1x1 if oct_conv_on else norm_conv1x1
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        norm_func = _BatchNorm2d if oct_conv_on else nn.BatchNorm2d
        act_func = _ReLU if oct_conv_on else nn.ReLU

        self.conv1 = conv1x1(inplanes, planes, type="first" if type == "first" else "normal")
        self.bn1 = norm_func(planes)
        self.relu1 = act_func(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride, type="last" if type == "last" else "normal")
        if type == "last":
            conv1x1 = norm_conv1x1
            norm_func = nn.BatchNorm2d
            act_func = nn.ReLU
        self.bn2 = norm_func(planes)
        self.relu2 = act_func(inplace=True)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_func(planes * self.expansion)
        self.relu3 = act_func(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if isinstance(out, (tuple, list)):
            assert len(out) == len(identity) and len(out) == 2
            out = (out[0] + identity[0], out[1] + identity[1])
        else:
            out += identity
        out = self.relu3(out)

        return out


class OctaveEncoder(nn.Module):
    def __init__(self, block, layers, input_channel=3):
        super(OctaveEncoder, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], type="first")
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, type="last")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, type="normal"):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or type == 'first':
            norm_func = nn.BatchNorm2d if type == "last" else _BatchNorm2d
            downsample = nn.Sequential(
                oct_conv1x1(self.inplanes, planes * block.expansion, stride, type=type),
                norm_func(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, type=type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, oct_conv_on=type != "last"))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]


octave_encoders = {
    'octave18': {
        'encoder': OctaveEncoder,
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [2, 2, 2, 2],
        },
    },

    'octave34': {
        'encoder': OctaveEncoder,
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [3, 4, 6, 3],
        },
    },

    'octave50': {
        'encoder': OctaveEncoder,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
        },
    },

    'octave101': {
        'encoder': OctaveEncoder,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
        },
    },

    'octave152': {
        'encoder': OctaveEncoder,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 8, 36, 3],
        },
    },
}