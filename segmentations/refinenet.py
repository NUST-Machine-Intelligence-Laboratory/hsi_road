import torch
import torch.nn as nn

import torch.nn.functional as F
from segmentations import EncoderDecoder, get_encoder


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self,
                    '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes, out_planes, stride=1, bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


class RefineNetDecoder(torch.nn.Module):

    def __init__(self, encoder_channels, decoder_channels, final_channels=2, agg_channels=256):
        super(RefineNetDecoder, self).__init__()
        self.ec = encoder_channels
        self.dc = decoder_channels
        self.agg = agg_channels
        self.do = nn.Dropout(p=0.2)
        self.p_ims1d2_outl1_dimred = conv1x1(self.ec[0], self.dc[0], bias=False)
        self.mflow_conv_g1_pool = self._make_crp(self.dc[0], self.dc[0], 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(self.dc[0], self.agg, bias=False)

        self.p_ims1d2_outl2_dimred = conv1x1(self.ec[1], self.dc[1], bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(self.dc[1], self.dc[1], bias=False)
        self.mflow_conv_g2_pool = self._make_crp(self.dc[1], self.dc[1], 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(self.dc[1], self.agg, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(self.ec[2], self.dc[2], bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(self.dc[2], self.dc[2], bias=False)
        self.mflow_conv_g3_pool = self._make_crp(self.dc[2], self.dc[2], 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(self.dc[2], self.agg, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(self.ec[3], self.dc[3], bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(self.dc[3], self.dc[3], bias=False)
        self.mflow_conv_g4_pool = self._make_crp(self.dc[3], self.agg, 4)

        self.clf_conv = nn.Conv2d(self.agg, final_channels, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def forward(self, x):

        l4, l3, l2, l1, _ = x

        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = F.relu(x4, inplace=True)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3, inplace=True)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2, inplace=True)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1, inplace=True)
        x1 = self.mflow_conv_g4_pool(x1)
        x0 = self.clf_conv(x1)

        out = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        return out


class RefineNet(EncoderDecoder):
    def __init__(self, encoder_name='resnet50', in_channel_nb=3, classes_nb=2):
        encoder, out_shapes = get_encoder(encoder_name, input_channel=in_channel_nb)
        decoder = RefineNetDecoder(
            encoder_channels=out_shapes,
            decoder_channels=(512, 256, 256, 256, 256),
            final_channels=classes_nb,
        )

        super().__init__(encoder, decoder)

        self.name = 'refinenet-{}'.format(encoder_name)


if __name__=='__main__':
    test_im = torch.randn([1, 3, 160, 160], dtype=torch.float32).to('cuda:0')
    model = RefineNet(encoder_name='resnet50', in_channel_nb=3, classes_nb=2)
    model.to('cuda:0')
    output = model(test_im)