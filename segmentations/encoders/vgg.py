import torch
import torch.nn as nn
from torchvision.models.vgg import cfgs


class VGGEncoder(nn.Module):
    def __init__(self, config, input_channel=3, batch_norm=False):
        super(VGGEncoder, self).__init__()
        self.batch_norm = batch_norm
        self.in_channels = input_channel
        self.sep = 0

        self.layer0 = self._make_layer(config)
        self.layer1 = self._make_layer(config)
        self.layer2 = self._make_layer(config)
        self.layer3 = self._make_layer(config)
        self.layer4 = self._make_layer(config)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, config):
        seq = list()
        start = self.sep
        for v in config[start:]:
            self.sep = self.sep + 1
            if v == 'M':
                seq += [nn.MaxPool2d(kernel_size=2, stride=2)]
                break
            else:
                conv2d = nn.Conv2d(self.in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    seq += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    seq += [conv2d, nn.ReLU(inplace=True)]
                self.in_channels = v
        return nn.Sequential(*seq)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]


vgg_encoder = {
    'vgg11': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 256, 128, 64),
        'params': {
            "config": cfgs['A'],
            "batch_norm": False,
        },
    },
    "vgg11_bn": {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 256, 128, 64),
        'params': {
            "config": cfgs['A'],
            "batch_norm": True,
        },
    },
    'vgg13': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 256, 128, 64),
        'params': {
            'config': cfgs['B'],
            'batch_norm': False,
        },
    },
    'vgg13_bn': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 256, 128, 64),
        'params': {
            'config': cfgs['B'],
            'batch_norm': True,
        },
    },
    'vgg16': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 256, 128, 64),
        'params': {
            'config': cfgs['D'],
            'batch_norm': False,
        },
    },
    'vgg16_bn': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 256, 128, 64),
        'params': {
            'config': cfgs['D'],
            'batch_norm': True,
        },
    },
    'vgg19': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 256, 128, 64),
        'params': {
            'config': cfgs['E'],
            'batch_norm': False,
        },
    },
    'vgg19_bn': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 256, 128, 64),
        'params': {
            'config': cfgs['E'],
            'batch_norm': True,
        },
    },
}

if __name__=='__main__':
    model = VGGEncoder(cfgs['D'], input_channel=3)
    print(model)
    a = torch.randn([1, 3, 32, 32], dtype=torch.float32, device='cuda:0')
    model.to('cuda:0')
    b = model(a)





