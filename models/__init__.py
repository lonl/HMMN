from __future__ import absolute_import

from .featureExtract import *
from .determiner import *
from .rank import *
from .determiner_test import *



__model_factory = {
    'resnet50': ResNet50,
    'featureExtract': featureExtract,
    'determiner': determiner,
    'determiner_test': determiner_test,
    'rank' : rank
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)