import torch
import torch.nn as nn


class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetEncoderV1b(nn.Module):

    def __init__(self, block, layers, input_channel=3, norm_layer=nn.BatchNorm2d):
        self.inplanes = 128
        super(ResNetEncoderV1b, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, 2, 1, bias=False),
            norm_layer(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            norm_layer(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)

        return [x4, x3, x2, x1, x0]


resnetv1b_encoders = {
    'resnetv1b50': {
        'encoder': ResNetEncoderV1b,
        'out_shapes': (2048, 1024, 512, 256, 128),
        'params': {
            'block': BottleneckV1b,
            'layers': [3, 4, 6, 3],
        },
    },

    'resnetv1b101': {
        'encoder': ResNetEncoderV1b,
        'out_shapes': (2048, 1024, 512, 256, 128),
        'params': {
            'block': BottleneckV1b,
            'layers': [3, 4, 23, 3],
        },
    },

    'resnetv1b152': {
        'encoder': ResNetEncoderV1b,
        'out_shapes': (2048, 1024, 512, 256, 128),
        'params': {
            'block': BottleneckV1b,
            'layers': [3, 8, 36, 3],
        },
    },
}