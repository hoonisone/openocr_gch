from openocr.openrec.modeling.decoders import build_decoder
from torch import nn


class NewGTCDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        gtc_decoder,
        ctc_decoder,
        detach=True,
        infer_gtc=False,
        out_channels=0,
        **kwargs,
    ):
        super(NewGTCDecoder, self).__init__()
        self.out_channels = out_channels
        self.detach = detach
        self.infer_gtc = infer_gtc
        if infer_gtc:
            gtc_decoder['out_channels'] = out_channels['gtc_num']
            ctc_decoder['out_channels'] = out_channels['ctc_num']
            gtc_decoder['in_channels'] = in_channels
            ctc_decoder['in_channels'] = in_channels
            self.gtc_decoder = build_decoder(gtc_decoder)
        else:
            ctc_decoder['in_channels'] = in_channels
            ctc_decoder['out_channels'] = out_channels
        self.ctc_decoder = build_decoder(ctc_decoder)

    def forward(self, x, data=None):
        ctc_pred = self.ctc_decoder(x.detach() if self.detach else x,
                                    data=data['ctc_label'])
        if self.training or self.infer_gtc:
            gtc_pred = self.gtc_decoder(x.flatten(2).transpose(1, 2),
                                        data=data['gtc_label'])
            return {'gtc_pred': gtc_pred, 'ctc_pred': ctc_pred}
        else:
            return ctc_pred


class NewGTCDecoderTwo(nn.Module):

    def __init__(
        self,
        in_channels,
        gtc_decoder,
        ctc_decoder,
        infer_gtc=False,
        out_channels=0,
        **kwargs,
    ):
        super(NewGTCDecoderTwo, self).__init__()
        self.infer_gtc = infer_gtc
        self.out_channels = out_channels
        # gtc_decoder['out_channels'] = out_channels[0]
        # ctc_decoder['out_channels'] = out_channels[1]
        gtc_decoder['out_channels'] = out_channels['gtc_num']
        ctc_decoder['out_channels'] = out_channels['ctc_num']
        gtc_decoder['in_channels'] = in_channels
        ctc_decoder['in_channels'] = in_channels
        self.gtc_decoder = build_decoder(gtc_decoder)
        self.ctc_decoder = build_decoder(ctc_decoder)

    def forward(self, x, data=None, return_feats=False):
        x_ctc, x_gtc = x

        if return_feats:
            ctc_feats, ctc_pred = self.ctc_decoder(
                x_ctc, data=data, return_features=True)

            if self.training or self.infer_gtc:
                gtc_feats, gtc_pred = self.gtc_decoder(
                    x_gtc.flatten(2).transpose(1, 2),
                    data=data,
                    return_features=True)
                return (
                    {'gtc_feats': gtc_feats, 'ctc_feats': ctc_feats},
                    {'gtc_pred': gtc_pred, 'ctc_pred': ctc_pred},
                )

            return {'ctc_feats': ctc_pred}, {'ctc_pred': ctc_pred}

        ctc_pred = self.ctc_decoder(x_ctc, data=data)

        if self.training or self.infer_gtc:
            gtc_pred = self.gtc_decoder(
                x_gtc.flatten(2).transpose(1, 2), data=data)
            return {'gtc_pred': gtc_pred, 'ctc_pred': ctc_pred}
        return ctc_pred
