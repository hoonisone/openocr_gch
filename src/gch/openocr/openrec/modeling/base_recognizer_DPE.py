import torch
from torch import nn
import random

from openocr.openrec.modeling.decoders import build_decoder
from openocr.openrec.modeling.encoders import build_encoder
from openocr.openrec.modeling.transforms import build_transform
from openocr.openrec.modeling import MODULES

__all__ = ['BaseRecognizer_DPE']


class BaseRecognizer_DPE(nn.Module):

    def __init__(self, config):
        """the module for OCR.

        args:
            config (dict): the super parameters for module.
        """
        super(BaseRecognizer_DPE, self).__init__()
        in_channels = config.get('in_channels', 3)
        self.use_wd = config.get('use_wd', True)
        self.direction_mode = str(config.get('direction_mode', 'oracle')).lower()

        if self.direction_mode not in {'oracle', 'random', 'adaptive'}:
            raise ValueError(
                f"Unsupported direction_mode: {self.direction_mode}. "
                "Use 'oracle', 'random', or 'adaptive'."
            )
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

    @torch.jit.ignore(drop=False)
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
        direction = self._resolve_direction(x)


        if self.use_transform:
            x = self.transform(x)
        if self.use_encoder:
            x = self.encoder(x, direction=direction)
        if self.use_decoder:
            x = self.decoder(x, data=data, direction=direction)
        return x

    def _infer_direction_from_shape(self, h: int, w: int) -> str:
        # horizontal: 0, vertical: 1
        return "vertical" if h > w else "horizontal"

    def _sample_adaptive_direction(self, h: int, w: int) -> str:
        # x = h / w, p(vertical) = x^2 / (1 + x^2)
        if w <= 0:
            return "vertical"
        ratio = float(h) / float(w)
        ratio_sq = ratio * ratio
        p_vertical = ratio_sq / (1.0 + ratio_sq)
        return "vertical" if random.random() < p_vertical else "horizontal"

    def _resolve_direction(self, x: torch.Tensor) -> str:
        if x.dim() < 4:
            raise ValueError(f"Expected input with shape [B, C, H, W], got {x.shape}")
        h, w = int(x.shape[-2]), int(x.shape[-1])

        # Inference always follows oracle rule.
        if not self.training:
            return self._infer_direction_from_shape(h, w)

        if self.direction_mode == 'random':
            return random.choice(["vertical", "horizontal"])
        if self.direction_mode == 'adaptive':
            return self._sample_adaptive_direction(h, w)
        return self._infer_direction_from_shape(h, w)



MODULES["BaseRecognizer_DPE"] = "gch.openocr.openrec.modeling.base_recognizer_DPE"
