from openocr.openrec.modeling.decoders import build_decoder, class_to_module
from torch import nn




class GCHDecoder(nn.Module):
    def __init__(self, c_encoder, g_encoder, in_channels, **kwargs):
        super(GCHDecoder, self).__init__()
        c_encoder['in_channels'] = in_channels
        g_encoder['in_channels'] = in_channels
        self.c_encoder = build_decoder(c_encoder)
        self.g_encoder = build_decoder(g_encoder)

    def forward(self, x, data=None):
        

        c_pred = self.c_encoder(x, data['c_label'])
        g_pred = self.g_encoder(x, data['g_label'])
        return {'c_pred': c_pred, 'g_pred': g_pred}


class_to_module['GCHDecoder'] = 'gch.openocr.openrec.modeling.decoder.gch_decoder'