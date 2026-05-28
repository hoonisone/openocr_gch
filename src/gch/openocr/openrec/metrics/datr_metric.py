from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from openocr.openrec.metrics import MODULES, build_metric

_DIR_ALIASES: Dict[str, str] = {
    "horizontal": "right", "vertical": "down", "h": "right", "v": "down",
}


def _normalize_dir(name: str) -> str:
    return _DIR_ALIASES.get(name.lower().strip(), name.lower().strip())


class DATRScoreMetric(object):
    """Score regression metric. Returns max, mes, rmse."""

    def __init__(self, main_indicator: str = "mes", **kwargs):
        self.main_indicator = main_indicator
        self.eps = 1e-8
        self.reset()

    def __call__(self, pred_label, batch=None, training=False):
        pred, target = pred_label
        pred_t = torch.as_tensor(pred, dtype=torch.float32).reshape(-1)
        target_t = torch.as_tensor(
            target, dtype=torch.float32, device=pred_t.device).reshape(-1)
        diff = pred_t - target_t
        abs_diff = torch.abs(diff)
        sq_diff = torch.square(diff)

        max_metric = torch.mean(abs_diff)
        mes_metric = torch.mean(sq_diff)
        rmse_metric = torch.sqrt(mes_metric)

        self.sample_count += int(diff.numel())
        self.sum_abs_error += float(torch.sum(abs_diff))
        self.sum_sq_error += float(torch.sum(sq_diff))

        return {
            "max": float(max_metric),
            "mes": float(mes_metric),
            "rmse": float(rmse_metric),
        }

    def get_metric(self, training=False):
        max_metric = self.sum_abs_error / (self.sample_count + self.eps)
        mes_metric = self.sum_sq_error / (self.sample_count + self.eps)
        rmse_metric = mes_metric**0.5
        result = {
            "max": max_metric,
            "mes": mes_metric,
            "rmse": rmse_metric,
        }
        self.reset()
        return result

    def reset(self):
        self.sample_count = 0
        self.sum_abs_error = 0.0
        self.sum_sq_error = 0.0


class DATRMetric(object):
    """Wrapper metric for text quality + direction score quality."""

    def __init__(
        self,
        text_metric: Dict[str, Any],
        score_metric: Optional[Dict[str, Any]] = None,
        report_eval_result_fallback: Optional[Any] = None,
        fallback_on_get_metric: bool = False,
        main_indicator: str = "acc",
        is_filter: bool = False,
        ignore_space: bool = True,
        stream: bool = False,
        with_ratio: bool = False,
        max_len: int = 25,
        max_ratio: int = 4,
        **kwargs,
    ):
        text_metric = dict(text_metric)
        text_metric["main_indicator"] = main_indicator
        text_metric["is_filter"] = is_filter
        text_metric["ignore_space"] = ignore_space
        text_metric["stream"] = stream
        text_metric["with_ratio"] = with_ratio
        text_metric["max_len"] = max_len
        text_metric["max_ratio"] = max_ratio

        self.text_metric = build_metric(text_metric)
        self.main_indicator = f"text_metric.{self.text_metric.main_indicator}"

        self.score_metric = build_metric(score_metric) if score_metric is not None else None

        self.report_eval_result_fallback = report_eval_result_fallback
        self.fallback_on_get_metric = bool(fallback_on_get_metric)
        self.latest_batch_eval_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------
    def _extract_text_preds(self, pred_label: Any) -> Tuple[
            Any, Optional[Dict[str, Any]], Optional[Dict[str, torch.Tensor]]]:
        if not isinstance(pred_label, dict):
            return pred_label, None, None
        pred = pred_label.get("pred", pred_label.get("text_pred", None))
        pred_by_dir: Optional[Dict[str, Any]] = pred_label.get("pred_by_dir", None)
        idx_by_dir: Optional[Dict[str, torch.Tensor]] = pred_label.get("idx_by_dir", None)

        # Backward compat
        if pred_by_dir is None:
            pred_h = pred_label.get("pred_h", None)
            pred_v = pred_label.get("pred_v", None)
            idx_h = pred_label.get("idx_h", None)
            idx_v = pred_label.get("idx_v", None)
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

        return pred, pred_by_dir, idx_by_dir

    def _extract_direction_scores(
            self, pred_label: Any
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Union[str, List[str]]]]:
        if not isinstance(pred_label, dict):
            return None, None
        direction_scores: Optional[Dict[str, torch.Tensor]] = pred_label.get(
            "direction_scores", None)
        direction = pred_label.get("selected_direction", None)

        # Backward compat
        if direction_scores is None:
            h_score = pred_label.get("h_score", None)
            v_score = pred_label.get("v_score", None)
            if h_score is not None or v_score is not None:
                direction_scores = {}
                if h_score is not None:
                    direction_scores["right"] = h_score
                if v_score is not None:
                    direction_scores["down"] = v_score

        if not isinstance(direction, (str, list)):
            direction = None
        return direction_scores, direction

    # ------------------------------------------------------------------
    # Data splitting
    # ------------------------------------------------------------------
    def _split_data_by_index(self, data: Any, idx: torch.Tensor,
                             original_batch_size: Optional[int]) -> Any:
        if data is None:
            return None
        idx_np = idx.detach().cpu().numpy().astype(np.int64)
        if isinstance(data, torch.Tensor):
            if data.dim() == 0:
                return data
            if original_batch_size is not None and data.shape[0] == original_batch_size:
                return data.index_select(0, idx)
            return data
        if isinstance(data, np.ndarray):
            if data.ndim == 0:
                return data
            if original_batch_size is not None and data.shape[0] == original_batch_size:
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

    # ------------------------------------------------------------------
    # Score target helpers
    # ------------------------------------------------------------------
    def _as_tensor_like(self, value: Any, ref: torch.Tensor) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(device=ref.device, dtype=ref.dtype)
        if isinstance(value, (int, float)):
            return torch.tensor(float(value), device=ref.device, dtype=ref.dtype)
        return None

    def _extract_ned_target_from_eval(self,
                                      eval_result: Optional[Dict[str, Any]],
                                      ref: torch.Tensor) -> Optional[torch.Tensor]:
        if not isinstance(eval_result, dict):
            return None
        if "norm_edit_dis" not in eval_result:
            return None
        target = self._as_tensor_like(eval_result["norm_edit_dis"], ref)
        if target is None:
            return None
        if target.shape == ref.shape:
            return target
        if ref.dim() == 0:
            return target.reshape(()) if target.numel() == 1 else None
        if target.dim() == 0:
            return target.expand_as(ref)
        if target.dim() == 1 and target.shape[0] == ref.shape[0]:
            view_shape = (target.shape[0],) + (1,) * (ref.dim() - 1)
            return target.view(*view_shape).expand_as(ref)
        if target.shape[0] != ref.shape[0]:
            return None
        return target

    def _pick_selected_score(self, direction_scores: Dict[str, torch.Tensor],
                             direction: Optional[Union[str, List[str]]]
                             ) -> Optional[torch.Tensor]:
        dir_names = list(direction_scores.keys())
        if len(dir_names) == 0:
            return None
        if isinstance(direction, str):
            d = _normalize_dir(direction)
            return direction_scores.get(d, direction_scores.get(direction))
        if isinstance(direction, list):
            stacked = torch.stack([direction_scores[d] for d in dir_names], dim=-1)
            if stacked.shape[0] != len(direction):
                return None
            indices = []
            for d_sample in direction:
                nd = _normalize_dir(d_sample)
                if nd in dir_names:
                    indices.append(dir_names.index(nd))
                else:
                    return None
            idx_t = torch.tensor(indices, device=stacked.device, dtype=torch.long)
            return stacked.gather(-1, idx_t.unsqueeze(-1)).squeeze(-1)
        if direction is None and len(dir_names) == 1:
            return direction_scores[dir_names[0]]
        return None

    def _merge_eval_results(self, eval_by_dir: Dict[str, Optional[Dict[str, Any]]],
                            count_by_dir: Dict[str, int]) -> Dict[str, Any]:
        valid = {d: e for d, e in eval_by_dir.items() if e is not None}
        if len(valid) == 0:
            return {}
        if len(valid) == 1:
            return dict(next(iter(valid.values())))
        total = sum(count_by_dir.get(d, 0) for d in valid)
        if total <= 0:
            return {}
        all_keys: set = set()
        for e in valid.values():
            all_keys |= e.keys()
        merged: Dict[str, Any] = {}
        for key in all_keys:
            vals = [(valid[d].get(key), count_by_dir.get(d, 0)) for d in valid
                    if key in valid[d]]
            if all(isinstance(v, (int, float)) for v, _ in vals):
                merged[key] = sum(float(v) * n for v, n in vals if v is not None) / float(total)  # type: ignore[arg-type]
            else:
                merged[key] = vals[0][0]
        return merged

    def _build_score_target(self, pred_label: Any, selected_score: torch.Tensor,
                            direction: Optional[Union[str, List[str]]],
                            eval_result: Dict[str, Any]) -> Optional[torch.Tensor]:
        eval_main = eval_result.get("eval", None) if isinstance(eval_result, dict) else None
        eval_by_dir: Dict[str, Any] = eval_result.get(
            "eval_by_dir", {}) if isinstance(eval_result, dict) else {}

        if isinstance(direction, list) and isinstance(pred_label, dict):
            idx_by_dir = pred_label.get("idx_by_dir", None)
            if isinstance(idx_by_dir, dict):
                total = sum(idx.numel() for idx in idx_by_dir.values()
                            if torch.is_tensor(idx))
                if selected_score.dim() > 0 and total == selected_score.shape[0]:
                    target = selected_score.new_zeros(selected_score.shape)
                    for d, idx in idx_by_dir.items():
                        if not torch.is_tensor(idx) or idx.numel() == 0:
                            continue
                        branch_ref = selected_score.new_zeros(
                            (int(idx.numel()),) + tuple(selected_score.shape[1:]))
                        branch_eval = eval_by_dir.get(d, eval_main)
                        branch_target = self._extract_ned_target_from_eval(
                            branch_eval, branch_ref)
                        if branch_target is None:
                            return None
                        target.index_copy_(0, idx, branch_target)
                    return target

        if isinstance(direction, str):
            d = _normalize_dir(direction)
            preferred = eval_by_dir.get(d, eval_main)
        else:
            preferred = eval_main
        if preferred is None:
            preferred = eval_main
        if preferred is None and eval_by_dir:
            preferred = next(iter(eval_by_dir.values()))
        return self._extract_ned_target_from_eval(preferred, selected_score)

    # ------------------------------------------------------------------
    # Score metric runner
    # ------------------------------------------------------------------
    def _run_score_metric(self, pred_label: Any, eval_result: Dict[str, Any],
                          batch: Any, training: bool) -> Optional[Dict[str, Any]]:
        if self.score_metric is None:
            return None
        direction_scores, direction = self._extract_direction_scores(pred_label)
        if direction_scores is None or len(direction_scores) == 0:
            return None
        selected_score = self._pick_selected_score(direction_scores, direction)
        if selected_score is None:
            return None
        score_target = self._build_score_target(
            pred_label, selected_score, direction, eval_result)
        if score_target is None:
            return None
        return self.score_metric(
            (selected_score, score_target), batch, training=training)

    # ------------------------------------------------------------------
    # Report / get_metric helpers
    # ------------------------------------------------------------------
    def _safe_report_eval_result(self, eval_result: Dict[str, Any]):
        fn = self.report_eval_result_fallback
        if not callable(fn):
            return
        fn(eval_result)

    def _safe_get_metric(self, metric_obj: Any, training: bool):
        try:
            return metric_obj.get_metric(training=training)
        except TypeError:
            return metric_obj.get_metric()

    def _extract_sample_ned_from_metric_result(
            self, metric_result: Any) -> Optional[torch.Tensor]:
        if isinstance(metric_result, dict):
            value = metric_result.get("sample_ned", None)
            if value is not None:
                return torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        getter = getattr(self.text_metric, "get_last_sample_ned", None)
        if callable(getter):
            value = getter()
            if value is not None:
                return torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        return None

    def _build_training_report_payload(
            self,
            eval_main: Dict[str, Any],
            eval_by_dir: Dict[str, Optional[Dict[str, Any]]],
            idx_by_dir: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Optional[Dict[str, Any]]:
        sample_ned_main = self._extract_sample_ned_from_metric_result(eval_main)
        sample_ned_by_dir: Dict[str, torch.Tensor] = {}

        for d, branch_eval in eval_by_dir.items():
            if branch_eval is None:
                continue
            sample_ned_d = self._extract_sample_ned_from_metric_result(branch_eval)
            if sample_ned_d is None:
                continue
            sample_ned_by_dir[d] = sample_ned_d

        merged_sample_ned = sample_ned_main
        if merged_sample_ned is None and isinstance(idx_by_dir, dict):
            total = sum(int(idx.numel()) for idx in idx_by_dir.values()
                        if torch.is_tensor(idx))
            if total > 0:
                merged = torch.zeros(total, dtype=torch.float32)
                filled = 0
                for d, idx in idx_by_dir.items():
                    if not torch.is_tensor(idx) or idx.numel() == 0:
                        continue
                    v = sample_ned_by_dir.get(d, None)
                    if v is None or int(v.shape[0]) != int(idx.numel()):
                        continue
                    merged.index_copy_(0, idx.detach().cpu(), v.detach().cpu())
                    filled += int(idx.numel())
                if filled == total:
                    merged_sample_ned = merged

        if merged_sample_ned is None and len(sample_ned_by_dir) == 0:
            return None

        eval_payload: Dict[str, Any] = {}
        if merged_sample_ned is not None:
            eval_payload["sample_ned"] = merged_sample_ned

        eval_by_dir_payload: Dict[str, Dict[str, Any]] = {}
        for d, v in sample_ned_by_dir.items():
            eval_by_dir_payload[d] = {"sample_ned": v}

        payload: Dict[str, Any] = {
            "eval": eval_payload,
            "eval_by_dir": eval_by_dir_payload,
        }
        if "right" in eval_by_dir_payload:
            payload["eval_h"] = eval_by_dir_payload["right"]
        if "down" in eval_by_dir_payload:
            payload["eval_v"] = eval_by_dir_payload["down"]
        return payload

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------
    def __call__(self, pred_label, batch=None, training=False):
        pred, pred_by_dir, idx_by_dir = self._extract_text_preds(pred_label)

        eval_main: Optional[Dict[str, Any]] = None
        eval_by_dir: Dict[str, Optional[Dict[str, Any]]] = {}

        if pred_by_dir is not None and idx_by_dir is not None:
            original_batch_size = sum(
                int(idx.numel()) for idx in idx_by_dir.values()
                if torch.is_tensor(idx))
            count_by_dir: Dict[str, int] = {}
            for d, pred_d in pred_by_dir.items():
                idx = idx_by_dir.get(d)
                if pred_d is None or not torch.is_tensor(idx) or idx.numel() == 0:
                    eval_by_dir[d] = None
                    count_by_dir[d] = 0
                    continue
                batch_d = self._split_data_by_index(batch, idx, original_batch_size)
                res = self.text_metric(pred_d, batch_d, training=training)
                eval_by_dir[d] = res if isinstance(res, dict) else None
                count_by_dir[d] = int(idx.numel())
            eval_main = self._merge_eval_results(eval_by_dir, count_by_dir)
        elif pred is not None:
            main_result = self.text_metric(pred, batch, training=training)
            if isinstance(main_result, dict):
                eval_main = main_result
        else:
            generic_result = self.text_metric(pred_label, batch, training=training)
            if isinstance(generic_result, dict):
                eval_main = generic_result

        if eval_main is None:
            eval_main = {}

        result: Dict[str, Any] = dict(eval_main)
        result["eval"] = eval_main
        result["eval_by_dir"] = eval_by_dir

        # Backward compat
        result["eval_h"] = eval_by_dir.get("right")
        result["eval_v"] = eval_by_dir.get("down")

        score_result = self._run_score_metric(pred_label, result, batch, training)
        if score_result is not None:
            result["score"] = score_result

        if training:
            self.latest_batch_eval_result = result
            report_payload = self._build_training_report_payload(
                eval_main=eval_main, eval_by_dir=eval_by_dir, idx_by_dir=idx_by_dir)
            self._safe_report_eval_result(report_payload
                                          if report_payload is not None else result)
        return result

    def get_metric(self, training=False):
        result = {}
        for k, v in self._safe_get_metric(self.text_metric, training).items():
            result[k] = v

        if self.score_metric is not None:
            for k, v in self._safe_get_metric(self.score_metric, training).items():
                result[f"score_{k}"] = v

        return result


MODULES[
    "DATRMetric"
] = "gch.openocr.openrec.metrics.datr_metric"
MODULES[
    "DATRScoreMetric"
] = "gch.openocr.openrec.metrics.datr_metric"
