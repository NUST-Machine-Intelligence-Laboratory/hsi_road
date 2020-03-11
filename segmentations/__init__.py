from .common import EncoderDecoder
from .common import Conv2dReLU
from .encoders import get_encoder

from .fcn import FCN
from .unet import Unet
from .pspnet import PSPNet
from .refinenet import RefineNet

__all__ = ['EncoderDecoder', 'get_encoder', 'Conv2dReLU']


def get_model(model_name, configs):
    model = None
    if model_name == 'unet':
        model = Unet(**configs)
    elif model_name == 'fcn':
        model = FCN(**configs)
    elif model_name == 'psp':
        model = PSPNet(**configs)
    elif model_name == 'refinenet':
        model = RefineNet(**configs)
    return model
