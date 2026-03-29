from .gch_label_encode import GCHLabelEncode


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
            out[key] = _extract_by_keep_keys(data[key], child)
        return out
    raise TypeError(
        'keep_keys node must be None, dict, or '
        f'list[str]/tuple[str], got {type(keep_keys)}')


class HierarchyKeepKeys:

    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        return _extract_by_keep_keys(data, self.keep_keys)


from openocr.openrec.preprocess import MODULE_MAPPING
MODULE_MAPPING['HierarchyKeepKeys'] = 'gch.openocr.openrec.preprocess'