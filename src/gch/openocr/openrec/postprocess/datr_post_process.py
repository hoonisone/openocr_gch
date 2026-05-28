from typing import Any, Dict, Optional

import numpy as np
import torch

from openocr.openrec.postprocess import build_post_process, module_mapping

module_mapping[
    'DATRPostProcess'
] = 'gch.openocr.openrec.postprocess.datr_post_process'


class DATRPostProcess(object):
    """Apply text postprocess on DATR output while preserving extras."""

    def __init__(self,
                 text_postprocess,
                 character_dict_path=None,
                 use_space_char=True,
                 **kwargs):
        text_postprocess = dict(text_postprocess)
        text_postprocess['character_dict_path'] = character_dict_path
        text_postprocess['use_space_char'] = use_space_char
        self.text_postprocess = build_post_process(text_postprocess)

    def _split_data_by_index(self, data: Any, idx: torch.Tensor,
                             original_batch_size: int) -> Any:
        if data is None:
            return None
        idx_np = idx.detach().cpu().numpy().astype(np.int64)
        if isinstance(data, torch.Tensor):
            if data.dim() == 0:
                return data
            if data.shape[0] == original_batch_size:
                return data.index_select(0, idx)
            return data
        if isinstance(data, np.ndarray):
            if data.ndim == 0:
                return data
            if data.shape[0] == original_batch_size:
                return data[idx_np]
            return data
        if isinstance(data, list):
            return [self._split_data_by_index(v, idx, original_batch_size)
                    for v in data]
        if isinstance(data, tuple):
            return tuple(self._split_data_by_index(v, idx, original_batch_size)
                         for v in data)
        if isinstance(data, dict):
            return {k: self._split_data_by_index(v, idx, original_batch_size)
                    for k, v in data.items()}
        return data

    def __call__(self, preds, batch=None, *args, **kwargs):
        if not isinstance(preds, dict):
            return self.text_postprocess(preds, batch=batch, *args, **kwargs)

        result = dict(preds)

        pred = preds.get('pred', None)
        pred_by_dir: Optional[Dict[str, Any]] = preds.get('pred_by_dir', None)
        idx_by_dir: Optional[Dict[str, torch.Tensor]] = preds.get('idx_by_dir', None)

        # Backward compat: check old keys if new keys absent
        if pred_by_dir is None:
            pred_h = preds.get('pred_h', None)
            pred_v = preds.get('pred_v', None)
            idx_h = preds.get('idx_h', None)
            idx_v = preds.get('idx_v', None)
            if pred_h is not None or pred_v is not None:
                pred_by_dir = {}
                idx_by_dir_tmp: Dict[str, Any] = {}
                if pred_h is not None:
                    pred_by_dir["right"] = pred_h
                    idx_by_dir_tmp["right"] = idx_h
                if pred_v is not None:
                    pred_by_dir["down"] = pred_v
                    idx_by_dir_tmp["down"] = idx_v
                idx_by_dir = idx_by_dir_tmp  # type: ignore[assignment]

        if pred is not None:
            result['pred'] = self.text_postprocess(pred, batch=batch, *args, **kwargs)

        if pred_by_dir is not None:
            original_batch_size: Optional[int] = None
            if isinstance(idx_by_dir, dict):
                original_batch_size = sum(
                    int(idx.numel()) for idx in idx_by_dir.values()
                    if torch.is_tensor(idx))

            processed_by_dir: Dict[str, Any] = {}
            for d, pred_d in pred_by_dir.items():
                if pred_d is None:
                    processed_by_dir[d] = None
                    continue
                batch_d = batch
                if (original_batch_size is not None
                        and isinstance(idx_by_dir, dict)
                        and torch.is_tensor(idx_by_dir.get(d))
                        and idx_by_dir[d].numel() > 0):
                    batch_d = self._split_data_by_index(
                        batch, idx_by_dir[d], original_batch_size)
                processed_by_dir[d] = self.text_postprocess(
                    pred_d, batch=batch_d, *args, **kwargs)
            result['pred_by_dir'] = processed_by_dir

            # Backward compat output
            if 'right' in processed_by_dir:
                result['pred_h'] = processed_by_dir['right']
            if 'down' in processed_by_dir:
                result['pred_v'] = processed_by_dir['down']

        return result

    def get_character_num(self):
        if hasattr(self.text_postprocess, 'get_character_num'):
            return self.text_postprocess.get_character_num()
        if hasattr(self.text_postprocess, 'character'):
            return len(self.text_postprocess.character)
        return {}
