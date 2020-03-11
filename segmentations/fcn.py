import torch
import torch.nn.functional as F

from segmentations import EncoderDecoder
from segmentations import get_encoder


class _FCNHead(torch.nn.Module):
    def __init__(self, in_channels, channels, norm_layer=torch.nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


class FCNDecoder(torch.nn.Module):
    def __init__(self, encoder_channels, final_channels):
        super(FCNDecoder, self).__init__()
        self.in_channels = encoder_channels
        self.final_channels = final_channels
        self.score_x4 = torch.nn.Conv2d(self.in_channels[0], final_channels, 1)
        self.score_x3 = torch.nn.Conv2d(self.in_channels[1], final_channels, 1)
        self.score_x2 = torch.nn.Conv2d(self.in_channels[2], final_channels, 1)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x4, x3, x2, _, _ = x

        score4 = self.score_x4(x4)
        score3 = self.score_x3(x3)
        score2 = self.score_x2(x2)
        score = F.interpolate(score4, score3.size()[2:], mode='bilinear', align_corners=True)
        score += score3
        score = F.interpolate(score, score2.size()[2:], mode='bilinear', align_corners=True)
        score += score2
        out = F.interpolate(score, scale_factor=8, mode='bilinear', align_corners=True)
        return out


class FCN(EncoderDecoder):
    def __init__(self, encoder_name='resnet50', in_channel_nb=3, classes_nb=2):
        encoder, out_shapes = get_encoder(encoder_name, input_channel=in_channel_nb)
        decoder = FCNDecoder(
            encoder_channels=out_shapes,
            final_channels=classes_nb
        )

        super().__init__(encoder, decoder)

        self.name = 'fcn-{}'.format(encoder_name)


if __name__=='__main__':
    test_im = torch.randn([1, 3, 160, 160], dtype=torch.float32).to('cuda:0')
    model = FCN(encoder_name='resnet50', in_channel_nb=3, classes_nb=2)
    model.to('cuda:0')
    output = model(test_im)
