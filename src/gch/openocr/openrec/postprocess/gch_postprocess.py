from openocr.openrec.postprocess import build_post_process, module_mapping


module_mapping['GCHPostProcess'] = 'gch.openocr.openrec.postprocess.gch_postprocess'

class GCHPostProcess(object):
    def __init__(self, c_postprocess, g_postprocess, **kwargs):
        self.c_postprocess = build_post_process(c_postprocess)
        self.g_postprocess = build_post_process(g_postprocess)

    def __call__(self, preds, batch=None, *args, **kwargs):
        # 원래 코드는 batch 맨 앞에 이미지가 있으나 ['c_label'] 등을 선택하면 한 칸씩 밀림
        # 따라서 idx를 맞추기 위해 아무 값이나 이미지 대신 맨 앞에 추가함함
        c_pred = self.c_postprocess(preds['c_pred'], batch=["temp_image"]+batch['c_label'], *args, **kwargs)
        g_pred = self.g_postprocess(preds['g_pred'], batch=["temp_image"]+batch['g_label'], *args, **kwargs)
        return {'c_pred': c_pred, 'g_pred': g_pred}

    
    def get_character_num(self):
        return {
            'c_num': self.c_postprocess.get_character_num(),
            'g_num': self.g_postprocess.get_character_num(),
        }