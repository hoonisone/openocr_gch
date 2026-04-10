from torch import nn
from openocr.openrec.losses import build_loss, name_to_module




class GCHLoss(nn.Module):
    def __init__(self, c_loss, g_loss, 
            c_weight:float = 1.0, 
            g_weight:float = 1.0, 
            use_c:bool = True, 
            use_g:bool = True, 
            **kwargs
        ):
        super(GCHLoss, self).__init__()
        self.c_loss = build_loss(c_loss)
        self.g_loss = build_loss(g_loss)
        self.use_c = use_c
        self.use_g = use_g

        total_weight = c_weight + g_weight

        self.c_weight = c_weight/total_weight
        self.g_weight = g_weight/total_weight

    def forward(self, predicts, batch):
        result = {}
        loss = 0
        if self.use_c:
            # [None] 은 batch 에 맨 앞에 Image가 있다고 가정하기 때문문

            assert isinstance(batch, dict), "이 모듈은 batch가 dict인 형태만 지원함"

            c_loss = self.c_loss(predicts['c_pred'], batch['c_label'])
            result['c_loss'] = c_loss
            # for k, v in c_loss.items():
            #     result[f'c_loss.{k}'] = v
            loss += c_loss['loss']*self.c_weight
                
        if self.use_g:
            g_loss = self.g_loss(predicts['g_pred'], batch['g_label'])
            result['g_loss'] = g_loss
            # for k, v in g_loss.items():
            #     result[f'g_loss.{k}'] = v
            loss += g_loss['loss']*self.g_weight

        result['loss'] = loss

        return result

class QualityWrapperLoss(nn.Module):
    def __init__(self, inner_loss,
            inner_weight:float = 0.95, 
            quality_weight:float = 0.05, 
            infer_quality:bool = True,
            detach:bool = False,
            **kwargs
        ):
        super(QualityWrapperLoss, self).__init__()
        self.decoder_loss = build_loss(inner_loss)
        self.infer_quality = infer_quality
        self.inner_weight = inner_weight
        self.quality_weight = quality_weight
        self.detach = detach
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        import torch.nn.functional as F
        # self.loss_f = lambda x, y: F.mse_loss(x, y)*0.1 + F.l1_loss(x, y)*0.9
        self.loss_f = F.smooth_l1_loss
    
        print("detach:", self.detach)

    def forward(self, predicts, batch=None):
        inner_loss = self.decoder_loss(predicts['inner_pred'], batch['inner_label'])
        
        if self.infer_quality:
            if self.detach:
                quality = predicts['quality_pred'].detach().squeeze()
            else:
                quality = predicts['quality_pred'].squeeze()
            dis_loss = self.loss_f(quality, batch['quality_label']['quality'])

            loss = {
                "inner_loss": inner_loss,
                "quality_loss": dis_loss
            }
            loss['loss'] = inner_loss['loss']*self.inner_weight + dis_loss*self.quality_weight
        
        else:
            loss = {
                "inner_loss": inner_loss,
            }
            loss['loss'] = loss['inner_loss']['loss']*self.inner_weight

        return loss

# class DistanceHeadLoss(nn.Module):
#     def __init__(self, dis_head_loss, **kwargs):
#         super(DistanceHeadLoss, self).__init__()
#         self.dis_head_loss = build_loss(dis_head_loss)

#     def forward(self, predicts, batch):
#         return self.dis_head_loss(predicts['distance_pred'], batch['distance_label'])

name_to_module['GCHLoss'] = 'gch.openocr.openrec.losses.gch_loss'
name_to_module['QualityWrapperLoss'] = 'gch.openocr.openrec.losses.gch_loss'