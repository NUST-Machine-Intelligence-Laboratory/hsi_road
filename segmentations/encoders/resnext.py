'''
New for ResNeXt:
1. Wider bottleneck
2. Add group for conv2
'''
import torch
import torch.nn as nn
import math

__all__ = ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101',

           'resnext152']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, input_channel=3, num_classes=1000, num_group=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_group)
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_group, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x4, x3, x2, x1, x0]

resnext_encoders = {
    'resnext': {
        'encoder': ResNeXt,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
        },
    },
}


if __name__=='__main__':
    model = ResNeXt(Bottleneck, [3, 4, 6, 3])
    a = torch.randn([4, 3, 64, 64], dtype=torch.float32)
    x4, x3, x2, x1, x0 = model(a)
    print(x4.size(), x3.size(), x2.size(), x1.size(), x0.size())




