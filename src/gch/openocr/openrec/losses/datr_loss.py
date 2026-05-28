from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from openocr.openrec.losses import build_loss, name_to_module

_DIR_ALIASES: Dict[str, str] = {
    "horizontal": "right", "vertical": "down", "h": "right", "v": "down",
}


def _normalize_dir(name: str) -> str:
    return _DIR_ALIASES.get(name.lower().strip(), name.lower().strip())


class DATRScoreLoss(nn.Module):
    """Generic score regression loss for direction-quality prediction."""

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        reduction: str = "mean",
        beta: float = 1.0,
        **kwargs,
    ):
        super(DATRScoreLoss, self).__init__()
        self.loss_type = str(loss_type).lower()
        self.reduction = reduction
        self.beta = float(beta)
        valid = {"l1", "smooth_l1", "mse"}
        if self.loss_type not in valid:
            raise ValueError(
                f"Unsupported loss_type: {self.loss_type}. "
                f"Use one of {sorted(valid)}.")

    def _parse_inputs(self, pred: Any, target: Any):
        if isinstance(pred, dict):
            if target is None:
                target = pred.get("ned_target", None)
            pred = pred.get("selected_score", pred.get("score", None))
        if pred is None or target is None:
            raise ValueError("DATRScoreLoss requires pred and target.")
        if not torch.is_tensor(pred):
            pred = torch.as_tensor(pred, dtype=torch.float32)
        if not torch.is_tensor(target):
            target = torch.as_tensor(target,
                                     dtype=pred.dtype,
                                     device=pred.device)
        target = target.to(device=pred.device, dtype=pred.dtype)
        return pred, target

    def forward(self, pred, target=None):
        pred, target = self._parse_inputs(pred, target)
        if self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == "mse":
            loss = F.mse_loss(pred, target, reduction=self.reduction)
        else:
            loss = F.smooth_l1_loss(pred,
                                    target,
                                    beta=self.beta,
                                    reduction=self.reduction)
        return {"loss": loss}


class DATRLoss(nn.Module):
    """Joint loss for text prediction and direction-score regression."""

    def __init__(
        self,
        text_loss: Dict[str, Any],
        score_loss: Optional[Dict[str, Any]] = None,
        text_weight: float = 1.0,
        score_weight: float = 1.0,
        reset_eval_result_after_forward: bool = True,
        **kwargs,
    ):
        super(DATRLoss, self).__init__()
        self.text_loss = build_loss(text_loss)
        self.score_loss = build_loss(score_loss) if score_loss is not None else None

        self.text_weight = float(text_weight)
        self.score_weight = float(score_weight)
        self.reset_eval_result_after_forward = bool(reset_eval_result_after_forward)

        self.eval_result: Optional[Any] = None

    def report_eval_result(self, eval_result: Any):
        self.eval_result = eval_result

    # ------------------------------------------------------------------
    # Text prediction extraction
    # ------------------------------------------------------------------
    def _extract_text_prediction(self, predicts: Any) -> Any:
        if isinstance(predicts, dict):
            if "pred" in predicts:
                return predicts["pred"]
            if "ctc_pred" in predicts or "gtc_pred" in predicts:
                return predicts
        return predicts

    # ------------------------------------------------------------------
    # Direction / score extraction
    # ------------------------------------------------------------------
    def _extract_selected_direction(
            self, predicts: Any) -> Optional[Union[str, List[str]]]:
        if not isinstance(predicts, dict):
            return None
        direction = predicts.get("selected_direction", None)
        if isinstance(direction, (str, list)):
            return direction
        return None

    def _extract_direction_scores(
            self, predicts: Any) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(predicts, dict):
            return None
        ds: Optional[Dict[str, torch.Tensor]] = predicts.get("direction_scores", None)
        if ds is not None:
            return ds
        # Backward compat
        h_score = predicts.get("h_score", None)
        v_score = predicts.get("v_score", None)
        if h_score is not None or v_score is not None:
            result: Dict[str, torch.Tensor] = {}
            if h_score is not None:
                result["right"] = h_score
            if v_score is not None:
                result["down"] = v_score
            return result
        return None

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

    # ------------------------------------------------------------------
    # NED target extraction
    # ------------------------------------------------------------------
    def _extract_ned_target(self, eval_result: Any) -> Optional[torch.Tensor]:
        if eval_result is None:
            return None
        if torch.is_tensor(eval_result):
            return eval_result.float()
        if isinstance(eval_result, (list, tuple)):
            return torch.as_tensor(eval_result, dtype=torch.float32)
        if isinstance(eval_result, (int, float)):
            return torch.tensor(float(eval_result), dtype=torch.float32)
        if isinstance(eval_result, dict):
            for key in [
                    "ned", "NED", "sample_ned", "instance_ned",
                    "direction_ned", "mean_ned", "norm_edit_dis",
            ]:
                if key in eval_result:
                    value = eval_result[key]
                    if torch.is_tensor(value):
                        return value.float()
                    if isinstance(value, (int, float)):
                        return torch.tensor(float(value), dtype=torch.float32)
        return None

    def _extract_ned_targets_from_eval(
            self, eval_result: Any
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        target_by_dir: Dict[str, Optional[torch.Tensor]] = {}
        if isinstance(eval_result, dict):
            eval_by_dir = eval_result.get("eval_by_dir", {})
            if isinstance(eval_by_dir, dict):
                for d, e in eval_by_dir.items():
                    target_by_dir[d] = self._extract_ned_target(e)
            # Backward compat
            if "eval_h" in eval_result and "right" not in target_by_dir:
                target_by_dir["right"] = self._extract_ned_target(
                    eval_result["eval_h"])
            if "eval_v" in eval_result and "down" not in target_by_dir:
                target_by_dir["down"] = self._extract_ned_target(
                    eval_result["eval_v"])
            target_main = self._extract_ned_target(eval_result.get("eval", None))
            if target_main is None:
                target_main = self._extract_ned_target(eval_result)
            return target_main, target_by_dir
        return self._extract_ned_target(eval_result), {}

    # ------------------------------------------------------------------
    # Data splitting
    # ------------------------------------------------------------------
    def _split_data_by_index(self, data: Any, idx: torch.Tensor,
                             original_batch_size: int) -> Any:
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            if data.dim() == 0:
                return data
            if data.shape[0] == original_batch_size:
                return data.index_select(0, idx)
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
    # Loss output normalization
    # ------------------------------------------------------------------
    def _as_loss_output(self, loss_output: Any) -> Dict[str, Any]:
        if isinstance(loss_output, dict):
            if "loss" in loss_output:
                return loss_output
            return {"loss": next(iter(loss_output.values()))}
        if torch.is_tensor(loss_output):
            return {"loss": loss_output}
        raise ValueError("Loss output must be Tensor or dict with 'loss'.")

    # ------------------------------------------------------------------
    # Score target builder
    # ------------------------------------------------------------------
    def _coerce_target_like(self, target: Optional[torch.Tensor],
                            ref: torch.Tensor) -> Optional[torch.Tensor]:
        if target is None:
            return None
        out = target.to(device=ref.device, dtype=ref.dtype)
        if out.shape == ref.shape:
            return out
        if ref.dim() == 0:
            return out.reshape(()) if out.numel() == 1 else None
        if out.dim() == 0:
            return out.expand_as(ref)
        if out.dim() == 1 and out.shape[0] == ref.shape[0]:
            view_shape = (out.shape[0],) + (1,) * (ref.dim() - 1)
            return out.view(*view_shape).expand_as(ref)
        if out.shape[0] != ref.shape[0]:
            return None
        return out

    def _build_score_target(
            self,
            predicts: Any,
            direction: Optional[Union[str, List[str]]],
            selected_score: torch.Tensor,
            target_main: Optional[torch.Tensor],
            target_by_dir: Dict[str, Optional[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if isinstance(direction, list) and isinstance(predicts, dict):
            idx_by_dir = predicts.get("idx_by_dir", None)
            if isinstance(idx_by_dir, dict):
                total = sum(idx.numel() for idx in idx_by_dir.values()
                            if torch.is_tensor(idx))
                if selected_score.dim() > 0 and total == selected_score.shape[0]:
                    main_target = (target_main.to(device=selected_score.device,
                                                  dtype=selected_score.dtype)
                                   if target_main is not None else None)
                    target = selected_score.new_zeros(selected_score.shape)
                    for d, idx in idx_by_dir.items():
                        if not torch.is_tensor(idx) or idx.numel() == 0:
                            continue
                        branch_ref = selected_score.new_zeros(
                            (int(idx.numel()),) + tuple(selected_score.shape[1:]))

                        # 1) Prefer branch-wise sample target (already split-order).
                        branch_target = self._coerce_target_like(
                            target_by_dir.get(d, None), branch_ref)

                        # 2) Fallback to main sample target sliced by branch indices.
                        if branch_target is None and main_target is not None:
                            if (main_target.dim() > 0
                                    and main_target.shape[0] == selected_score.shape[0]):
                                branch_target = self._coerce_target_like(
                                    main_target.index_select(0, idx), branch_ref)

                        # 3) Final fallback: scalar/main broadcast behavior.
                        if branch_target is None:
                            branch_target = self._coerce_target_like(
                                main_target, branch_ref)
                        if branch_target is None:
                            return None
                        target.index_copy_(0, idx, branch_target)
                    return target

        if isinstance(direction, str):
            d = _normalize_dir(direction)
            preferred = target_by_dir.get(d, target_main)
        else:
            preferred = target_main
        if preferred is None:
            preferred = target_main
        if preferred is None and target_by_dir:
            preferred = next(
                (v for v in target_by_dir.values() if v is not None), None)
        return self._coerce_target_like(preferred, selected_score)

    # ------------------------------------------------------------------
    # Text loss
    # ------------------------------------------------------------------
    def _compute_text_loss(self, predicts: Any, batch: Any) -> Dict[str, Any]:
        text_pred = self._extract_text_prediction(predicts)
        if text_pred is not None:
            text_loss_output = self._as_loss_output(self.text_loss(text_pred, batch))
            text_loss_output["text_loss"] = text_loss_output["loss"]
            return text_loss_output

        if not isinstance(predicts, dict):
            raise ValueError("predicts must be dict when merged pred is None.")

        pred_by_dir: Optional[Dict[str, Any]] = predicts.get("pred_by_dir", None)
        idx_by_dir: Optional[Dict[str, torch.Tensor]] = predicts.get("idx_by_dir", None)

        # Backward compat
        if pred_by_dir is None:
            pred_h = predicts.get("pred_h", None)
            pred_v = predicts.get("pred_v", None)
            idx_h = predicts.get("idx_h", None)
            idx_v = predicts.get("idx_v", None)
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

        if pred_by_dir is None or idx_by_dir is None:
            raise ValueError("No text prediction found.")
        if not isinstance(idx_by_dir, dict):
            raise ValueError("idx_by_dir must be a dict of tensors.")

        original_batch_size = sum(
            int(idx.numel()) for idx in idx_by_dir.values()
            if torch.is_tensor(idx))
        result: Dict[str, Any] = {}
        weighted_loss: Optional[torch.Tensor] = None
        total_count = 0

        for d, pred_d in pred_by_dir.items():
            idx = idx_by_dir.get(d)
            if pred_d is None or not torch.is_tensor(idx) or idx.numel() == 0:
                continue
            batch_d = self._split_data_by_index(batch, idx, original_batch_size)
            out_d = self._as_loss_output(self.text_loss(pred_d, batch_d))
            loss_d = out_d["loss"]
            result[f"text_loss_{d}"] = loss_d
            contrib = loss_d * idx.numel()
            weighted_loss = contrib if weighted_loss is None else weighted_loss + contrib
            total_count += int(idx.numel())

        if weighted_loss is None or total_count == 0:
            raise ValueError("Split text loss got no valid branch.")
        text_loss = weighted_loss / float(total_count)
        result["text_loss"] = text_loss
        result["loss"] = text_loss
        return result

    # ------------------------------------------------------------------
    # Score loss
    # ------------------------------------------------------------------
    def _compute_score_loss(self, predicts: Any, batch: Any) -> Optional[Dict[str, Any]]:
        if self.score_loss is None:
            return None

        direction_scores = self._extract_direction_scores(predicts)
        direction = self._extract_selected_direction(predicts)
        target_main, target_by_dir = self._extract_ned_targets_from_eval(
            self.eval_result)

        if direction_scores is None or len(direction_scores) == 0:
            return None

        selected_score = self._pick_selected_score(direction_scores, direction)
        if selected_score is None:
            return None
        ned_target = self._build_score_target(
            predicts, direction, selected_score, target_main, target_by_dir)
        if ned_target is None:
            return None

        score_loss_output: Any = None
        try:
            score_loss_output = self.score_loss(selected_score, ned_target)
        except TypeError:
            payload = {
                "selected_score": selected_score,
                "direction_scores": direction_scores,
                "selected_direction": direction,
                "ned_target": ned_target,
            }
            try:
                score_loss_output = self.score_loss(payload)
            except TypeError:
                score_loss_output = self.score_loss(payload, batch)

        try:
            return self._as_loss_output(score_loss_output)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, predicts, batch):
        text_loss_output = self._compute_text_loss(predicts, batch)

        result: Dict[str, Any] = text_loss_output
        total_loss = self.text_weight * text_loss_output["loss"]

        try:
            score_loss_output = self._compute_score_loss(predicts, batch)
        finally:
            if self.reset_eval_result_after_forward:
                self.eval_result = None

        if score_loss_output is not None:
            score_loss = score_loss_output["loss"]
            result["score_loss"] = score_loss
            total_loss = total_loss + self.score_weight * score_loss

        result["loss"] = total_loss
        return result


name_to_module[
    "DATRLoss"
] = "gch.openocr.openrec.losses.datr_loss"
name_to_module[
    "DATRScoreLoss"
] = "gch.openocr.openrec.losses.datr_loss"
