import string
from collections import defaultdict

import numpy as np
from rapidfuzz.distance import Levenshtein

from openocr.openrec.metrics import MODULES
from gch.openocr.openrec.preprocess.gch_label_encode import KoreanTransfomer
from .f1_score_tool import build_char_bin_summary, load_char_train_count

MODULES["RecMetricWithF1"] = "gch.openocr.openrec.metrics.rec_metric_with_f1"


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
            s_start, s_list[s_i][1:-1] if s_i < s_n - 1 else s_list[s_i][1:]
        )
        s_new += s_start
    return s_new, sum(conf_list) / bs


class RecMetricWithF1(object):
    def __init__(
        self,
        main_indicator="acc",
        is_filter=False,
        is_lower=True,
        ignore_space=True,
        stream=False,
        with_ratio=False,
        max_len=25,
        max_ratio=4,
        return_per_char=True, 
        g2c:bool=False, # 문자열 평가 전에 g2c 변환을 적용할 지
        c2g:bool=False, # 문자열 평가 전에 c2g 변환을 적용할 지
        char_train_count_path:str="/home/resources/gch/1_datasets/1_AIHUB/2_split_horizontal_clean_80:10:10/1_train/1_main___id_14/char_count.txt", # 글자 등장 빈도 파일 경로
        char_bin_edges:list[int]=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 60, 100, 300, 600, 1000, 3000], # 글자 등장 빈도 구간 값
        **kwargs
    ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.is_lower = is_lower
        self.ignore_space = ignore_space
        self.stream = stream
        self.eps = 1e-5
        self.with_ratio = with_ratio
        self.max_len = max_len
        self.max_ratio = max_ratio
        self.return_per_char = return_per_char
        self.reset()

        self.g2c = g2c
        self.c2g = c2g
        self.char_train_count_path = char_train_count_path
        self.char_bin_edges = list(char_bin_edges) if char_bin_edges is not None else []
        self._char_train_count_cache = None

        assert not (self.g2c and self.c2g), "g2c and c2g cannot be True at the same time"
        if self.g2c or self.c2g:
            self.korean_transformer = KoreanTransfomer() 

    def _normalize_text(self, text):
        text = "".join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text)
        )
        return text

    def _nw_score(self, s1, s2, match_score=2, mismatch_score=-1, gap_score=-1):
        prev = [j * gap_score for j in range(len(s2) + 1)]
        for i, ch1 in enumerate(s1, start=1):
            curr = [i * gap_score] + [0] * len(s2)
            for j, ch2 in enumerate(s2, start=1):
                diag = prev[j - 1] + (match_score if ch1 == ch2 else mismatch_score)
                up = prev[j] + gap_score
                left = curr[j - 1] + gap_score
                curr[j] = max(diag, up, left)
            prev = curr
        return prev

    def _nw_align(self, s1, s2, match_score=2, mismatch_score=-1, gap_score=-1):
        n, m = len(s1), len(s2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = i * gap_score
        for j in range(1, m + 1):
            dp[0][j] = j * gap_score

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                diag = dp[i - 1][j - 1] + (
                    match_score if s1[i - 1] == s2[j - 1] else mismatch_score
                )
                up = dp[i - 1][j] + gap_score
                left = dp[i][j - 1] + gap_score
                dp[i][j] = max(diag, up, left)

        aligned = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                diag_score = dp[i - 1][j - 1] + (
                    match_score if s1[i - 1] == s2[j - 1] else mismatch_score
                )
                if dp[i][j] == diag_score:
                    aligned.append((s1[i - 1], s2[j - 1]))
                    i -= 1
                    j -= 1
                    continue
            if i > 0 and dp[i][j] == dp[i - 1][j] + gap_score:
                aligned.append((s1[i - 1], None))
                i -= 1
            else:
                aligned.append((None, s2[j - 1]))
                j -= 1
        aligned.reverse()
        return aligned

    def _hirschberg_align(self, s1, s2, match_score=2, mismatch_score=-1, gap_score=-1):
        if len(s1) == 0:
            return [(None, c2) for c2 in s2]
        if len(s2) == 0:
            return [(c1, None) for c1 in s1]
        if len(s1) == 1 or len(s2) == 1:
            return self._nw_align(
                s1, s2, match_score=match_score, mismatch_score=mismatch_score, gap_score=gap_score
            )

        split = len(s1) // 2
        left_score = self._nw_score(
            s1[:split], s2, match_score=match_score, mismatch_score=mismatch_score, gap_score=gap_score
        )
        right_score = self._nw_score(
            s1[split:][::-1],
            s2[::-1],
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_score,
        )

        split_j = 0
        best = None
        for j in range(len(s2) + 1):
            score = left_score[j] + right_score[len(s2) - j]
            if best is None or score > best:
                best = score
                split_j = j

        left_alignment = self._hirschberg_align(
            s1[:split],
            s2[:split_j],
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_score,
        )
        right_alignment = self._hirschberg_align(
            s1[split:],
            s2[split_j:],
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_score,
        )
        return left_alignment + right_alignment

    def _update_char_confusion(self, pred, target):
        alignment = self._hirschberg_align(pred, target)
        self.total_char_events += len(alignment)

        tp = 0
        fp = 0
        fn = 0
        for pred_c, target_c in alignment:
            if pred_c is not None:
                self._char_seen.add(pred_c)
            if target_c is not None:
                self._char_seen.add(target_c)

            if pred_c is not None and target_c is not None and pred_c == target_c:
                self.per_char_confusion[pred_c]["tp"] += 1
                tp += 1
            elif pred_c is None and target_c is not None:
                self.per_char_confusion[target_c]["fn"] += 1
                fn += 1
            elif pred_c is not None and target_c is None:
                self.per_char_confusion[pred_c]["fp"] += 1
                fp += 1
            else:
                self.per_char_confusion[pred_c]["fp"] += 1
                self.per_char_confusion[target_c]["fn"] += 1
                fp += 1
                fn += 1

        self.char_tp += tp
        self.char_fp += fp
        self.char_fn += fn

    def _char_summary(self):
        precision = self.char_tp / (self.char_tp + self.char_fp + self.eps)
        recall = self.char_tp / (self.char_tp + self.char_fn + self.eps)
        char_f1 = (2.0 * precision * recall) / (precision + recall + self.eps)
        return {
            "char_tp": self.char_tp,
            "char_fp": self.char_fp,
            "char_fn": self.char_fn,
            "char_precision": precision,
            "char_recall": recall,
            "char_f1": char_f1,
        }

    def _per_char_summary(self):
        if not self.return_per_char:
            return {}
        char_report = {}
        for ch in sorted(self._char_seen):
            counts = self.per_char_confusion[ch]
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            tn = self.total_char_events - tp - fp - fn
            precision = tp / (tp + fp + self.eps)
            recall = tp / (tp + fn + self.eps)
            f1 = (2.0 * precision * recall) / (precision + recall + self.eps)
            char_report[ch] = {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        return {"per_char_confusion": char_report}

    def _load_char_train_count(self):
        if self._char_train_count_cache is not None:
            return self._char_train_count_cache

        char_count = load_char_train_count(self.char_train_count_path)
        self._char_train_count_cache = char_count
        return char_count

    def _char_bin_summary(self):
        char_count = self._load_char_train_count()
        return build_char_bin_summary(
            per_char_confusion=self.per_char_confusion,
            total_char_events=self.total_char_events,
            char_train_count=char_count,
            char_bin_edges=self.char_bin_edges,
            eps=self.eps,
        )

    def __call__(self, pred_label, batch=None, training=False, *args, **kwargs):
        if self.with_ratio and not training:
            return self.eval_all_metric(pred_label, batch)
        
        if training:
            return self.train_mode_eval_metric(pred_label)
        return self.eval_metric(pred_label)

    def train_mode_eval_metric(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0

        for (pred, pred_conf), (target, _) in zip(preds, labels):
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
            
            if self.g2c:
                pred = ''.join(self.korean_transformer.g2c(pred))
                target = ''.join(self.korean_transformer.g2c(target))
            elif self.c2g:
                pred = ''.join(self.korean_transformer.c2g(pred))
                target = ''.join(self.korean_transformer.c2g(target))

            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1

            all_num += 1

        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis

        return {
            "acc": correct_num / (all_num + self.eps),
            "norm_edit_dis": 1 - norm_edit_dis / (all_num + self.eps),
        }


    def eval_metric(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        batch_char_tp = 0
        batch_char_fp = 0
        batch_char_fn = 0

        for (pred, pred_conf), (target, _) in zip(preds, labels):
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
            
            if self.g2c:
                pred = ''.join(self.korean_transformer.g2c(pred))
                target = ''.join(self.korean_transformer.g2c(target))
            elif self.c2g:
                pred = ''.join(self.korean_transformer.c2g(pred))
                target = ''.join(self.korean_transformer.c2g(target))

            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1

            prev_tp = self.char_tp
            prev_fp = self.char_fp
            prev_fn = self.char_fn
            self._update_char_confusion(pred, target)
            batch_char_tp += self.char_tp - prev_tp
            batch_char_fp += self.char_fp - prev_fp
            batch_char_fn += self.char_fn - prev_fn
            all_num += 1

        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        batch_precision = batch_char_tp / (batch_char_tp + batch_char_fp + self.eps)
        batch_recall = batch_char_tp / (batch_char_tp + batch_char_fn + self.eps)
        batch_char_f1 = (
            2.0 * batch_precision * batch_recall / (batch_precision + batch_recall + self.eps)
        )

        return {
            "acc": correct_num / (all_num + self.eps),
            "norm_edit_dis": 1 - norm_edit_dis / (all_num + self.eps),
            "char_f1": batch_char_f1,
        }

    def eval_all_metric(self, pred_label, batch=None, *args, **kwargs):
        if self.with_ratio:
            ratio = batch[-1]
        preds, labels = pred_label
        correct_num = 0
        correct_num_real = 0
        correct_num_lower = 0
        correct_num_ignore_space = 0
        correct_num_ignore_space_lower = 0
        correct_num_ignore_space_symbol = 0
        all_num = 0
        norm_edit_dis = 0.0
        each_len_num = [0 for _ in range(self.max_len)]
        each_len_correct_num = [0 for _ in range(self.max_len)]
        each_len_norm_edit_dis = [0 for _ in range(self.max_len)]
        each_ratio_num = [0 for _ in range(self.max_ratio)]
        each_ratio_correct_num = [0 for _ in range(self.max_ratio)]
        each_ratio_norm_edit_dis = [0 for _ in range(self.max_ratio)]
        batch_char_tp = 0
        batch_char_fp = 0
        batch_char_fn = 0

        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.stream:
                assert len(labels) == 1
                pred, _ = stream_match(preds)
            if pred == target:
                correct_num_real += 1

            if pred.lower() == target.lower():
                correct_num_lower += 1

            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if pred == target:
                correct_num_ignore_space += 1

            if pred.lower() == target.lower():
                correct_num_ignore_space_lower += 1

            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            if pred == target:
                correct_num_ignore_space_symbol += 1

            if self.is_lower:
                pred = pred.lower()
                target = target.lower()
            dis = Levenshtein.normalized_distance(pred, target)
            norm_edit_dis += dis
            ratio_i = ratio[all_num] - 1 if ratio[all_num] < self.max_ratio else self.max_ratio - 1
            len_i = max(0, min(self.max_len, len(target)) - 1)
            if pred == target:
                correct_num += 1
                each_len_correct_num[len_i] += 1
                each_ratio_correct_num[ratio_i] += 1
            each_len_num[len_i] += 1
            each_len_norm_edit_dis[len_i] += dis

            each_ratio_num[ratio_i] += 1
            each_ratio_norm_edit_dis[ratio_i] += dis

            prev_tp = self.char_tp
            prev_fp = self.char_fp
            prev_fn = self.char_fn
            self._update_char_confusion(pred, target)
            batch_char_tp += self.char_tp - prev_tp
            batch_char_fp += self.char_fp - prev_fp
            batch_char_fn += self.char_fn - prev_fn
            all_num += 1

        self.correct_num += correct_num
        self.correct_num_real += correct_num_real
        self.correct_num_lower += correct_num_lower
        self.correct_num_ignore_space += correct_num_ignore_space
        self.correct_num_ignore_space_lower += correct_num_ignore_space_lower
        self.correct_num_ignore_space_symbol += correct_num_ignore_space_symbol
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        self.each_len_num = self.each_len_num + np.array(each_len_num)
        self.each_len_correct_num = self.each_len_correct_num + np.array(each_len_correct_num)
        self.each_len_norm_edit_dis = self.each_len_norm_edit_dis + np.array(each_len_norm_edit_dis)
        self.each_ratio_num = self.each_ratio_num + np.array(each_ratio_num)
        self.each_ratio_correct_num = self.each_ratio_correct_num + np.array(each_ratio_correct_num)
        self.each_ratio_norm_edit_dis = self.each_ratio_norm_edit_dis + np.array(
            each_ratio_norm_edit_dis
        )

        batch_precision = batch_char_tp / (batch_char_tp + batch_char_fp + self.eps)
        batch_recall = batch_char_tp / (batch_char_tp + batch_char_fn + self.eps)
        batch_char_f1 = (
            2.0 * batch_precision * batch_recall / (batch_precision + batch_recall + self.eps)
        )

        return {
            "acc": correct_num / (all_num + self.eps),
            "norm_edit_dis": 1 - norm_edit_dis / (all_num + self.eps),
            "char_f1": batch_char_f1,
        }

    def get_metric(self, training=False):
        if self.with_ratio and not training:
            return self.get_all_metric()
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        num_samples = self.all_num
        result = {
            "acc": acc,
            "norm_edit_dis": norm_edit_dis,
            "num_samples": num_samples,
        }
        if not training:
            result.update(self._char_summary())
            result.update(self._per_char_summary())
            result.update(self._char_bin_summary())
        self.reset()
        return result

    def get_all_metric(self):
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        acc_real = 1.0 * self.correct_num_real / (self.all_num + self.eps)
        acc_lower = 1.0 * self.correct_num_lower / (self.all_num + self.eps)
        acc_ignore_space = 1.0 * self.correct_num_ignore_space / (self.all_num + self.eps)
        acc_ignore_space_lower = 1.0 * self.correct_num_ignore_space_lower / (
            self.all_num + self.eps
        )
        acc_ignore_space_symbol = 1.0 * self.correct_num_ignore_space_symbol / (
            self.all_num + self.eps
        )

        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        num_samples = self.all_num
        each_len_acc = (self.each_len_correct_num / (self.each_len_num + self.eps)).tolist()
        each_len_norm_edit_dis = (
            1 - ((self.each_len_norm_edit_dis) / ((self.each_len_num) + self.eps))
        ).tolist()
        each_len_num = self.each_len_num.tolist()
        each_ratio_acc = (self.each_ratio_correct_num / (self.each_ratio_num + self.eps)).tolist()
        each_ratio_norm_edit_dis = (
            1 - ((self.each_ratio_norm_edit_dis) / ((self.each_ratio_num) + self.eps))
        ).tolist()
        each_ratio_num = self.each_ratio_num.tolist()

        result = {
            "acc": acc,
            "acc_real": acc_real,
            "acc_lower": acc_lower,
            "acc_ignore_space": acc_ignore_space,
            "acc_ignore_space_lower": acc_ignore_space_lower,
            "acc_ignore_space_symbol": acc_ignore_space_symbol,
            "acc_ignore_space_lower_symbol": acc,
            "each_len_num": each_len_num,
            "each_len_acc": each_len_acc,
            "each_len_norm_edit_dis": each_len_norm_edit_dis,
            "each_ratio_num": each_ratio_num,
            "each_ratio_acc": each_ratio_acc,
            "each_ratio_norm_edit_dis": each_ratio_norm_edit_dis,
            "norm_edit_dis": norm_edit_dis,
            "num_samples": num_samples,
        }
        result.update(self._char_summary())
        result.update(self._per_char_summary())
        result.update(self._char_bin_summary())
        self.reset()
        return result

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.correct_num_real = 0
        self.correct_num_lower = 0
        self.correct_num_ignore_space = 0
        self.correct_num_ignore_space_lower = 0
        self.correct_num_ignore_space_symbol = 0
        self.each_len_num = np.array([0 for _ in range(self.max_len)])
        self.each_len_correct_num = np.array([0 for _ in range(self.max_len)])
        self.each_len_norm_edit_dis = np.array([0.0 for _ in range(self.max_len)])
        self.each_ratio_num = np.array([0 for _ in range(self.max_ratio)])
        self.each_ratio_correct_num = np.array([0 for _ in range(self.max_ratio)])
        self.each_ratio_norm_edit_dis = np.array([0.0 for _ in range(self.max_ratio)])

        self.char_tp = 0
        self.char_fp = 0
        self.char_fn = 0
        self.total_char_events = 0
        self.per_char_confusion = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        self._char_seen = set()
