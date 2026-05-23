from torch import nn

from openocr.openrec.losses import build_loss


class NewGTCLoss(nn.Module):

    def __init__(
        self,
        gtc_loss,
        ctc_loss,
        gtc_weight=1.0,
        ctc_weight=0.1,
        zero_infinity=True,
        **kwargs,
    ):
        super(NewGTCLoss, self).__init__()
        ctc_loss['zero_infinity'] = zero_infinity
        self.ctc_loss = build_loss(ctc_loss)
        self.gtc_loss = build_loss(gtc_loss)
        self.gtc_weight = gtc_weight
        self.ctc_weight = ctc_weight

    def forward(self, predicts, batch):
        if isinstance(batch, dict):
            ctc_batch = batch['ctc_label']
            gtc_batch = batch['gtc_label']
        else:
            ctc_batch = [None] + batch[-2:]
            gtc_batch = [None] + batch[:-2]

        ctc_loss = self.ctc_loss(predicts['ctc_pred'], ctc_batch)

        gtc_loss = self.gtc_loss(predicts['gtc_pred'], gtc_batch)
        return {
            'loss': self.ctc_weight * ctc_loss['loss'] + self.gtc_weight *
            gtc_loss['loss'],
            'ctc_loss': ctc_loss,
            'gtc_loss': gtc_loss,
        }
