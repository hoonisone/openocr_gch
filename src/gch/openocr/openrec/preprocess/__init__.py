from .gch_label_encode import GCHLabelEncode
from openocr.tools.utils.logging import get_logger


only_one_log_print_flat = False # 중복이 많을 것 같아 한 번만 출력하기 위한 플래그

def _extract_by_keep_keys(data, keep_keys):
    """Extract values from data with the same tree shape as keep_keys.

    keep_keys supports:
      - None: stop traversal and return current data as-is
      - dict: values are dict or list[str]/tuple[str]
      - list[str]/tuple[str]: leaf keys to fetch from current data dict
    """
    if keep_keys is None:
        return data
    if isinstance(keep_keys, list):
        return [data[key] for key in keep_keys]
    if isinstance(keep_keys, tuple):
        return tuple(data[key] for key in keep_keys)
    if isinstance(keep_keys, dict):
        out = {}
        for key, child in keep_keys.items():
            if key not in data:
                if only_one_log_print_flat:
                    get_logger().warning(f"Key {key} not found in data, this log will be printed only once for 간결함, you need to check if more key not found in data is exists")
                    only_one_log_print_flat = True
                continue
            out[key] = _extract_by_keep_keys(data[key], child)
        return out
    raise TypeError(
        'keep_keys node must be None, dict, or '
        f'list[str]/tuple[str], got {type(keep_keys)}')


class HierarchyKeepKeys:

    def __init__(self, 
        keep_keys, 
        **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        return _extract_by_keep_keys(data, self.keep_keys)


from openocr.openrec.preprocess import MODULE_MAPPING

MODULE_MAPPING['HierarchyKeepKeys'] = 'gch.openocr.openrec.preprocess'
MODULE_MAPPING['NewGTCLabelEncode'] = (
    'gch.openocr.openrec.preprocess.new_gtc_label_encode')
MODULE_MAPPING['NewVisionLANLabelEncode'] = (
    'gch.openocr.openrec.preprocess.new_visionlan_label_encode')
MODULE_MAPPING['TilingAug'] = 'gch.openocr.openrec.preprocess.tiling_aug'
MODULE_MAPPING['HV_90Rotate'] = 'gch.openocr.openrec.preprocess.hv_rotate'
MODULE_MAPPING['Flip'] = 'gch.openocr.openrec.preprocess.rec_aug'