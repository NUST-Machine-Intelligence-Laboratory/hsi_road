"""Context Encoding for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import EncoderDecoder
from .encoders import get_encoder
from .fcn import _FCNHead


class Encoding(nn.Module):
    def __init__(self, d, k):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.d, self.k = d, k
        self.codewords = nn.Parameter(torch.zeros(k, d), requires_grad=True)
        self.scale = nn.Parameter(torch.zeros(k), requires_grad=True)
        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, x):
        # input X is a 4D tensor
        assert (x.size(1) == self.d)
        b, d = x.size(0), self.d
        if x.dim() == 3:
            # BxDxN -> BxNxD
            x = x.transpose(1, 2).contiguous()
        elif x.dim() == 4:
            # BxDxHxW -> Bx(HW)xD
            x = x.view(b, d, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        a = F.softmax(self.scale_l2(x, self.codewords, self.scale), dim=2)
        # aggregate
        e = self.aggregate(a, x, self.codewords)
        return e

    @staticmethod
    def scale_l2(x, c, s):
        s = s.view(1, 1, c.size(0), 1)
        x = x.unsqueeze(2).expand(x.size(0), x.size(1), c.size(0), c.size(1))
        c = c.unsqueeze(0).unsqueeze(0)
        sl = s * (x - c)
        sl = sl.pow(2).sum(3)
        return sl

    @staticmethod
    def aggregate(a, x, c):
        a = a.unsqueeze(3)
        x = x.unsqueeze(2).expand(x.size(0), x.size(1), c.size(0), c.size(1))
        c = c.unsqueeze(0).unsqueeze(0)
        e = a * (x - c)
        e = e.sum(1)
        return e


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            Encoding(d=in_channels, k=ncodes),
            nn.BatchNorm1d(ncodes),
            nn.ReLU(True),
            Mean(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class _EncHead(nn.Module):
    def __init__(self, in_channels, nclass, se_loss=True):
        super(_EncHead, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.encmodule = EncModule(512, nclass, ncodes=32, se_loss=se_loss)
        self.conv6 = nn.Sequential(
            nn.Dropout(0.1, False),
            nn.Conv2d(512, nclass, 1)
        )

    def forward(self, *inputs):
        # conv-bn-relu. output 512
        feat = self.conv5(inputs[-1])
        # encoding.
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)


class EncDecoder(nn.Module):
    def __init__(self, encoder_channels, final_channels, imsize):
        super(EncDecoder, self).__init__()
        self.head = _EncHead(encoder_channels[-1], final_channels)
        self.aux = _FCNHead(encoder_channels[-2], final_channels)
        self.imsize = imsize

    def forward(self, x):
        x4, x3, x2, x1, _ = x
        x = self.head(x1, x2, x3, x4)
        x[0] = F.interpolate(x[0], self.imsize, mode='bilinear', align_corners=True)
        auxout = self.aux(x2)
        auxout = F.interpolate(auxout, self.imsize, mode='bilinear', align_corners=True)
        x.append(auxout)
        return tuple(x)


class EncNet(EncoderDecoder):
    def __init__(self, encoder_name='resnet50V1b', in_channel_nb=3, classes_nb=2, aux=True, se_loss=True):
        encoder = get_encoder(encoder_name, input_channel=in_channel_nb)
        '''
        self.head = _EncHead(2048, classes_nb, se_loss=se_loss)
        if aux:
            self.auxlayer = _FCNHead(1024, classes_nb)
        '''
        super(EncoderDecoder, self).__init__(encoder, decoder)


    def forward(self, x):
        size = x.size()[2:]
        features = self.base_forward(x)

        x = list(self.head(*features))
        x[0] = F.interpolate(x[0], size, mode='bilinear', align_corners=True)
        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            x.append(auxout)
        return tuple(x)







