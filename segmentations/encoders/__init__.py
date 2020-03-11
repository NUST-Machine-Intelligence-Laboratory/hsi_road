from .vgg import vgg_encoder
from .resnet import resnet_encoders
from .densenet import densenet_encoders
from .dpn import dpn_encoders
from .senet import senet_encoders
from .octave import octave_encoders
from .cam import cam_encoders
from .mobilenetv3 import mobilenet_encoders
from .jpu import JPU


encoders = {}
encoders.update(vgg_encoder)
encoders.update(resnet_encoders)
encoders.update(densenet_encoders)
encoders.update(dpn_encoders)
encoders.update(senet_encoders)
encoders.update(octave_encoders)
encoders.update(cam_encoders)
encoders.update(mobilenet_encoders)


def get_encoder(name, input_channel=3):
    enc = encoders[name]['encoder']
    params = dict()
    print(encoders[name]['params'])
    params.update(encoders[name]['params'])
    params.update({'input_channel': input_channel})
    encoder = enc(**params)
    return encoder, encoders[name]['out_shapes']


def get_encoder_names():
    return list(encoders.keys())
