import torch
from .resnet import ResNetEncoder
from .resnet import BasicBlock
from .resnet import Bottleneck


class CAMModule(torch.nn.Module):
    def __init__(self, in_dim):
        super(CAMModule, self).__init__()
        self.chanel_in = in_dim
        self.gamma = torch.nn.Parameter(torch.ones([self.chanel_in, 1]), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.channel_norm = torch.nn.InstanceNorm2d(self.chanel_in)

    def forward(self, x):
        b, c, h, w = x.size()
        x_norm = self.channel_norm(x)
        proj_query = x_norm.view(b, c, -1)  # b c n
        proj_key = x_norm.view(b, c, -1).permute(0, 2, 1)  # b n c
        energy = torch.bmm(proj_query, proj_key)  # b c c , 通道之间的内积(夹角差异)
        energy_max = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        attention = self.softmax(self.gamma * (energy_max - energy))  # redistribute 聚合系数
        proj_value = x.view(b, c, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(b, c, h, w)

        out = out + x
        return out


class CamEncoder(ResNetEncoder):

    def __init__(self, block, layers, input_channel=3):
        super(CamEncoder, self).__init__(block=block, layers=layers, input_channel=input_channel)
        self.cam0 = CAMModule(input_channel)

    def forward(self, x):
        x0 = self.cam0(x)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]


cam_encoders = {
    'cam18': {
        'encoder': CamEncoder,
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [2, 2, 2, 2],
        },
    },

    'cam34': {
        'encoder': CamEncoder,
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [3, 4, 6, 3],
        },
    },

    'cam50': {
        'encoder': CamEncoder,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
        },
    },

    'cam101': {
        'encoder': CamEncoder,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
        },
    },

    'cam152': {
        'encoder': CamEncoder,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 8, 36, 3],
        },
    },
}
