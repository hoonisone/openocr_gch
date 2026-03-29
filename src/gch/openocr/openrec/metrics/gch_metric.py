from openocr.openrec.metrics import build_metric, MODULES

MODULES['RecGCHMetric'] = 'gch.openocr.openrec.metrics.gch_metric'


from openocr.tools.utils.logging import get_logger

class RecGCHMetric(object):
    def __init__(self, c_metric, g_metric):
        self.c_metric = build_metric(c_metric)
        self.g_metric = build_metric(g_metric)

    def __call__(self, pred_label, batch=None, training=False):
        c_metric = self.c_metric(pred_label['c_pred'], batch, training=training)
        g_metric = self.g_metric(pred_label['g_pred'], batch, training=training)
        return {
            'c_metric': c_metric,
            'g_metric': g_metric
        }

    def main_indicator(self)->str:
        return f"c_metric.{self.c_metric.main_indicator}"

    def get_metric(self):
        
        logger = get_logger()
        logger.info("metric 반환에서 c 만 반환중, 코드 수정 필요요")
        return self.c_metric.get_metric()
        return {
            'c_metric': self.c_metric.get_metric(),
            'g_metric': self.g_metric.get_metric()
        }