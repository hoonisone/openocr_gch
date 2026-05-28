import string
from typing import List, Optional

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from openocr.openrec.metrics import MODULES

MODULES["DATRRecMetric"] = "gch.openocr.openrec.metrics.datr_rec_metric"


def match_ss(ss1, ss2):
    s1_len = len(ss1)
    for c_i in range(s1_len):
        if ss1[c_i:] == ss2[:s1_len - c_i]:
            return ss2[s1_len - c_i:]
    return ss2


def stream_match(text):
    bs = len(text)
    s_list = []
    conf_list = []
    for s_conf in text:
        s_list.append(s_conf[0])
        conf_list.append(s_conf[1])
    s_n = bs
    s_start = s_list[0][:-1]
    s_new = s_start
    for s_i in range(1, s_n):
        s_start = match_ss(
            s_start, s_list[s_i][1:-1] if s_i < s_n - 1 else s_list[s_i][1:])
        s_new += s_start
    return s_new, sum(conf_list) / bs


class DATRRecMetric(object):
    """Rec metric with per-sample NED output for DATR score targets."""

    def __init__(self,
                 main_indicator="acc",
                 is_filter=False,
                 is_lower=True,
                 ignore_space=True,
                 stream=False,
                 with_ratio=False,
                 max_len=25,
                 max_ratio=4,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.is_lower = is_lower
        self.ignore_space = ignore_space
        self.stream = stream
        self.eps = 1e-5
        self.with_ratio = with_ratio
        self.max_len = max_len
        self.max_ratio = max_ratio
        self.latest_sample_ned: Optional[torch.Tensor] = None
        self.reset()

    def _normalize_text(self, text):
        text = "".join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text

    def _evaluate_pairs(self, preds, labels):
        correct_num = 0
        all_num = 0
        norm_edit_dis_sum = 0.0
        sample_ned_list: List[float] = []

        for (pred, _pred_conf), (target, _) in zip(preds, labels):
            if self.stream:
                assert len(labels) == 1
                pred, _ = stream_match(preds)
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            if self.is_lower:
                pred = pred.lower()
                target = target.lower()

            distance = float(Levenshtein.normalized_distance(pred, target))
            sample_ned = 1.0 - distance
            sample_ned_list.append(sample_ned)
            norm_edit_dis_sum += distance

            if pred == target:
                correct_num += 1
            all_num += 1

        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis_sum
        self.latest_sample_ned = torch.tensor(sample_ned_list, dtype=torch.float32)

        return {
            "acc": correct_num / (all_num + self.eps),
            "norm_edit_dis": 1 - norm_edit_dis_sum / (all_num + self.eps),
            # "sample_ned": self.latest_sample_ned,
        }

    def __call__(self, pred_label, batch=None, training=False, *args, **kwargs):
        if self.with_ratio and not training:
            return self.eval_all_metric(pred_label, batch)
        return self.eval_metric(pred_label)

    def eval_metric(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        return self._evaluate_pairs(preds, labels)

    def eval_all_metric(self, pred_label, batch=None, *args, **kwargs):
        # Keep compatibility with with_ratio mode, but include sample_ned as well.
        preds, labels = pred_label
        return self._evaluate_pairs(preds, labels)

    def get_last_sample_ned(self) -> Optional[torch.Tensor]:
        return self.latest_sample_ned

    def get_metric(self, training=False):
        if self.with_ratio and not training:
            return self.get_all_metric()
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        num_samples = self.all_num
        self.reset()
        return {
            "acc": acc,
            "norm_edit_dis": norm_edit_dis,
            "num_samples": num_samples,
        }

    def get_all_metric(self):
        # Return a compatible summary for with_ratio mode.
        return self.get_metric(training=False)

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0.0
        self.latest_sample_ned = None
