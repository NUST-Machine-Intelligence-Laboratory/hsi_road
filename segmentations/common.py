import torch


class Conv2dReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True, **batchnorm_params):

        super().__init__()

        layers = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not (use_batchnorm)),
            torch.nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, torch.nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderDecoder(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """return logits!"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
