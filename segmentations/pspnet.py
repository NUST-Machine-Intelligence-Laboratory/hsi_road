import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentations import EncoderDecoder, Conv2dReLU, get_encoder


class PyramidStage(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.stages = nn.ModuleList(
            [PyramidStage(in_channels, in_channels // len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes]
        )

    def forward(self, x):
        xs = [stage(x) for stage in self.stages] + [x]
        x = torch.cat(xs, dim=1)
        return x


class AUXModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = F.adaptive_max_pool2d(x, output_size=(1, 1))
        x = x.view(-1, x.size(1))
        x = self.linear(x)
        return x


class PSPDecoder(torch.nn.Module):
    def __init__(self, encoder_channels, final_channels=21, downsample_factor=8, use_batchnorm=True, psp_out_channels=512, aux_output=False, dropout=0.2):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.out_channels = self._get(encoder_channels)
        self.aux_output = aux_output
        self.dropout_factor = dropout
        self.psp = PSPModule(self.out_channels, sizes=(1, 2, 3, 6), use_bathcnorm=use_batchnorm)
        self.conv = Conv2dReLU(self.out_channels * 2, psp_out_channels, kernel_size=1, use_batchnorm=use_batchnorm)
        if self.dropout_factor:
            self.dropout = nn.Dropout2d(p=dropout)
        self.final_conv = nn.Conv2d(psp_out_channels, final_channels, kernel_size=(3, 3), padding=1)
        if self.aux_output:
            self.aux = AUXModule(self.out_channels, final_channels)

    def _get(self, xs):
        if self.downsample_factor == 4:
            return xs[3]
        elif self.downsample_factor == 8:
            return xs[2]
        elif self.downsample_factor == 16:
            return xs[1]
        else:
            raise ValueError('Downsample factor should bi in [4, 8, 16], got {}'.format(self.downsample_factor))

    def forward(self, x):
        features = self._get(x)
        x = self.psp(features)
        x = self.conv(x)
        if self.dropout_factor:
            x = self.dropout(x)
        x = self.final_conv(x)
        x = F.interpolate(x, scale_factor=self.downsample_factor, mode='bilinear', align_corners=True)

        if self.training and self.aux_output:
            aux = self.aux(features)
            x = [x, aux]

        return x


class PSPNet(EncoderDecoder):
    def __init__(
            self, encoder_name='resnet50', in_channel_nb=3, classes_nb=2,
            psp_in_factor=8, psp_out_channels=512, psp_use_batchnorm=True, psp_aux_output=False, dropout=0.2):

        encoder, out_shapes = get_encoder(encoder_name, input_channel=in_channel_nb)

        decoder = PSPDecoder(
            encoder_channels=out_shapes,
            downsample_factor=psp_in_factor,
            psp_out_channels=psp_out_channels,
            final_channels=classes_nb,
            dropout=dropout,
            aux_output=psp_aux_output,
            use_batchnorm=psp_use_batchnorm,
        )

        super().__init__(encoder, decoder)

        self.name = 'psp-{}'.format(encoder_name)


if __name__=='__main__':
    test_im = torch.randn([1, 3, 160, 160], dtype=torch.float32).to('cuda:0')
    model = PSPNet(encoder_name='vgg16', in_channel_nb=3, classes_nb=2)
    model.to('cuda:0')
    output = model(test_im)
