import torch
import torch.nn.functional as F

from segmentations import Conv2dReLU, EncoderDecoder, get_encoder


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = torch.nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class CenterBlock(DecoderBlock):
    def forward(self, x):
        return self.block(x)


class UnetDecoder(torch.nn.Module):
    def __init__(self, encoder_channels, decoder_channels, final_channels, use_batchnorm=True):
        super().__init__()

        in_channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            decoder_channels[3]
        ]
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)
        self.final_conv = torch.nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        return x


class Unet(EncoderDecoder):
    def __init__(self, encoder_name='resnet50', in_channel_nb=3, classes_nb=2):
        encoder, out_shapes = get_encoder(encoder_name, input_channel=in_channel_nb)
        decoder = UnetDecoder(
            encoder_channels=out_shapes,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=classes_nb,
            use_batchnorm=True
        )

        super().__init__(encoder, decoder)

        self.name = 'unet-{}'.format(encoder_name)


if __name__=='__main__':
    test_im = torch.randn([1, 3, 160, 160], dtype=torch.float32).to('cuda:0')
    model = Unet(encoder_name='resnet50', in_channel_nb=3, classes_nb=2)
    model.to('cuda:0')
    output = model(test_im)
