from openocr.openrec.postprocess import build_post_process, module_mapping
from rapidfuzz.distance import Levenshtein
import torch

module_mapping['GCHPostProcess'] = 'gch.openocr.openrec.postprocess.gch_postprocess'
module_mapping['QualityWrapperPostProcess'] = 'gch.openocr.openrec.postprocess.gch_postprocess'

class GCHPostProcess(object):
    def __init__(self, c_postprocess, g_postprocess, use_c:bool = True, use_g:bool = True, **kwargs):
        self.c_postprocess = build_post_process(c_postprocess)
        self.g_postprocess = build_post_process(g_postprocess)
        self.use_c = use_c
        self.use_g = use_g
        self.ensemble = GCHEnsemble()
        self.super_ensemble = SuperGCHEnsemble()

    def __call__(self, preds, batch=None, *args, **kwargs):
        # 원래 코드는 batch 맨 앞에 이미지가 있으나 ['c_label'] 등을 선택하면 한 칸씩 밀림
        # 따라서 idx를 맞추기 위해 아무 값이나 이미지 대신 맨 앞에 추가함함
        result = {}
        if self.use_c:
            result['c_pred'] = self.c_postprocess(preds['c_pred'], batch=batch['c_label'], *args, **kwargs)
        if self.use_g:
            result['g_pred'] = self.g_postprocess(preds['g_pred'], batch=batch['g_label'], *args, **kwargs)
        
        if self.use_c and "quality_pred" in result['c_pred']['pred'] and self.use_g and "quality_pred" in result['g_pred']['pred']:
            c_pred = result["c_pred"]["pred"]['inner_pred'][0]
            c_quality = result["c_pred"]["pred"]['quality_pred'][0]
            c_label = result["c_pred"]['pred']["inner_pred"][1]
            g_pred = result['g_pred']["pred"]['inner_pred'][0]
            g_quality = result['g_pred']["pred"]['quality_pred'][0]
            g_label = result['g_pred']["pred"]['inner_pred'][1]

            result['e_pred'] = self.ensemble(c_pred, c_quality, c_label, g_pred, g_quality, g_label)
            result['o_pred'] = self.super_ensemble(c_pred, c_quality, c_label, g_pred, g_quality, g_label)

        return result

    
    def get_character_num(self):
        result = {}
        result['c_num'] = self.c_postprocess.get_character_num()
        result['g_num'] = self.g_postprocess.get_character_num()
        
        return result

class GCHEnsemble(object):
    def __init__(self):
        self.korean_transformer = KoreanTransfomer()

    def __call__(self, c_pred, c_quality, c_label, g_pred, g_quality, g_label):
        preds = []
        labels = []
        for c_p, c_q, c_l, g_p, g_q, g_l in zip(c_pred, c_quality, c_label, g_pred, g_quality, g_label):
            if c_q > g_q: # 퀄리티가 더 높은 것을 선택
                _c_p, _ = c_p
                _c_p = ''.join(self.korean_transformer.c2g(_c_p))
                c_p = (_c_p, _)

                _c_l, _ = c_l
                _c_l = ''.join(self.korean_transformer.c2g(_c_l))
                c_l = (_c_l, _)
                preds.append(c_p)
                labels.append(c_l)
            else:
                preds.append(g_p)
                labels.append(g_l)
        return preds, labels

class SuperGCHEnsemble(object):
    def __init__(self):
        self.korean_transformer = KoreanTransfomer()

    def __call__(self, c_pred, c_quality, c_label, g_pred, g_quality, g_label):
        preds = []
        labels = []
        for c_p, c_q, c_l, g_p, g_q, g_l in zip(c_pred, c_quality, c_label, g_pred, g_quality, g_label):
            c_q = 1-Levenshtein.normalized_distance(c_pred[0], c_label[0])
            g_q = 1-Levenshtein.normalized_distance(g_pred[0], g_label[0])
            if c_q > g_q:
                _c_p, _ = c_p
                _c_p = ''.join(self.korean_transformer.c2g(_c_p))
                c_p = (_c_p, _)

                _c_l, _ = c_l
                _c_l = ''.join(self.korean_transformer.c2g(_c_l))
                c_l = (_c_l, _)

                preds.append(c_p)
                labels.append(c_l)
            else:
                preds.append(g_p)
                labels.append(g_l)
        return preds, labels

from gch.openocr.openrec.preprocess.gch_label_encode import KoreanTransfomer

class QualityWrapperPostProcess(object):
    def __init__(self, inner_postprocess, 
        character_dict_path=None,
        use_space_char=True,
        infer_quality:bool = True,
        **kwargs
    ):
        inner_postprocess['character_dict_path'] = character_dict_path
        inner_postprocess['use_space_char'] = use_space_char

        self.inner_postprocess = build_post_process(inner_postprocess, **kwargs)
        self.infer_quality = infer_quality
        
        self.korean_transformer = KoreanTransfomer()

    @property
    def character(self):
        return self.inner_postprocess.character

    def __call__(self, preds, batch=None, *args, **kwargs):
        result = {}
        result['inner_pred'] = self.inner_postprocess(preds['inner_pred'], batch=batch['inner_label'], *args, **kwargs)

        if self.infer_quality:
            result["quality_pred"] = self.process_head_pred(preds['quality_pred'], result['inner_pred'],batch=batch['quality_label'], *args, **kwargs)
        
        if batch and self.infer_quality:
            batch['quality_label']['quality'] = result["quality_pred"][1]

        return result

    def process_head_pred(self, preds, inner_pred, batch=None):
        
        pred_numpy = preds.detach().cpu().float().numpy()
        if batch is None:
            return pred_numpy

        device = preds.device
        quality_label = self.make_quality_label(inner_pred, device)
        
        return preds, quality_label
        
    def make_quality_label(self, postprocess_result, device):
        dis_labels = []
        for (pred, _), (label, _) in zip(*postprocess_result):
            pred = self.korean_transformer.c2g(pred)
            label = self.korean_transformer.c2g(label)
            quality = 1-Levenshtein.normalized_distance(pred, label)
            dis_labels.append(quality)

        dis_labels = torch.tensor(dis_labels, dtype=torch.float32, device=device)
        return dis_labels

    