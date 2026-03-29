from torch import nn
from openocr.openrec.losses import build_loss, name_to_module




class GCHLoss(nn.Module):
    def __init__(self, c_loss, g_loss, **kwargs):
        super(GCHLoss, self).__init__()
        self.c_loss = build_loss(c_loss)
        self.g_loss = build_loss(g_loss)

    def forward(self, predicts, batch):
        c_loss = self.c_loss(predicts['c_pred'], batch['c_label'])
        g_loss = self.g_loss(predicts['g_pred'], batch['g_label'])
        return {
            'loss': c_loss['loss'] + g_loss['loss'],
            'c_loss': c_loss['loss'],
            'g_loss': g_loss['loss']
        }


name_to_module['GCHLoss'] = 'gch.openocr.openrec.losses.gch_loss'