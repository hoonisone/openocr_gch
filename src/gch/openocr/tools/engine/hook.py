import torch.nn as nn
from importlib import import_module

__all__ = ['build_hook']

class_to_module = {
    'QualityOnHook': 'gch.openocr.tools.engine.hook',
}


def build_hook(config):

    module_name = config.pop('name')

    # Check if the class is defined in current module (e.g., GTCDecoder)
    # if module_name in globals():
    #     module_class = globals()[module_name]
    # else:
    if module_name not in class_to_module:
        raise ValueError(f'Unsupported decoder: {module_name}')
    module_str = class_to_module[module_name]
    # Dynamically import the module and get the class
    module = import_module(module_str, package=__package__)
    module_class = getattr(module, module_name)

    return module_class(**config)






class Hook:
    def before_epoch(self, trainer, epoch):
        pass


class QualityOnHook(Hook):
    def __init__(self, epoch):
        self.on_epoch = epoch
        self.done = False


    def before_epoch(self, trainer, epoch:int):
        if self.done is False and self.on_epoch <= epoch:
            print("Turn on Quality Option", epoch)
            trainer.model.decoder.c_decoder.ctc_decoder.infer_distance = True
            trainer.model.decoder.g_decoder.ctc_decoder.infer_distance = True
            trainer.loss_class.c_loss.ctc_loss.infer_quality = True
            trainer.loss_class.g_loss.ctc_loss.infer_quality = True
            trainer.post_process_class.c_postprocess.ctc_label_decode.infer_quality = True
            trainer.post_process_class.g_postprocess.ctc_label_decode.infer_quality = True
            trainer.eval_class.c_metric.ctc_metric.infer_quality = True
            trainer.eval_class.g_metric.ctc_metric.infer_quality = True
            self.done = True
