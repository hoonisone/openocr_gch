from openocr.openrec.modeling import MODULES
from openocr.openrec.modeling.base_recognizer import BaseRecognizer

# __all__ = ['GCHRecognizer']


from torch import nn
import torch

from openocr.openrec.modeling.transforms import build_transform
from openocr.openrec.modeling.encoders import build_encoder
from openocr.openrec.modeling.decoders import build_decoder

class GCHRecognizer(nn.Module):

    def __init__(self, config):
        """the module for OCR.

        args:
            config (dict): the super parameters for module.
        """
        super(GCHRecognizer, self).__init__()
        in_channels = config.get('in_channels', 3)
        self.use_wd = config.get('use_wd', True)
        # build transfrom,
        # for rec, transfrom can be TPS,None
        if 'Transform' not in config or config['Transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['Transform']['in_channels'] = in_channels
            self.transform = build_transform(config['Transform'])
            in_channels = self.transform.out_channels

        # build backbone
        if 'Encoder' not in config or config['Encoder'] is None:
            self.use_encoder = False
        else:
            self.use_encoder = True
            config['Encoder']['in_channels'] = in_channels
            self.encoder = build_encoder(config['Encoder'])
            in_channels = self.encoder.out_channels

        # build decoder
        if 'Decoder' not in config or config['Decoder'] is None:
            self.use_decoder = False
        else:
            self.use_decoder = True
            config['Decoder']['in_channels'] = in_channels
            self.decoder = build_decoder(config['Decoder'])

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.use_wd:
            if hasattr(self.encoder, 'no_weight_decay'):
                no_weight_decay = self.encoder.no_weight_decay()
            else:
                no_weight_decay = {}
            if hasattr(self.decoder, 'no_weight_decay'):
                no_weight_decay.update(self.decoder.no_weight_decay())
            return no_weight_decay
        else:
            return {}

    def forward(self, x, data=None):
        if self.use_transform:
            x = self.transform(x)
        if self.use_encoder:
            x = self.encoder(x)
        if self.use_decoder:
            x = self.decoder(x, data=data)
        return x


MODULES["GCHRecognizer"] = "gch.openocr.openrec.modeling.gch_recognizer"