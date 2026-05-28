from typing import Optional

import torch

from openocr.openrec.metrics import MODULES, build_metric

MODULES["DATRRecGTCMetric"] = "gch.openocr.openrec.metrics.datr_rec_metric_gtc"


class DATRRecGTCMetric(object):
    """GTC metric wrapper that exposes sample-wise NED from ctc_metric."""

    def __init__(
        self,
        ctc_metric,
        gtc_metric,
        main_indicator="acc",
        is_filter=False,
        ignore_space=True,
        stream=False,
        with_ratio=False,
        max_len=25,
        max_ratio=4,
        **kwargs,
    ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5

        ctc_metric = dict(ctc_metric)
        gtc_metric = dict(gtc_metric)

        ctc_metric["main_indicator"] = main_indicator
        ctc_metric["is_filter"] = is_filter
        ctc_metric["ignore_space"] = ignore_space
        ctc_metric["stream"] = stream
        ctc_metric["with_ratio"] = with_ratio
        ctc_metric["max_len"] = max_len
        ctc_metric["max_ratio"] = max_ratio

        gtc_metric["main_indicator"] = main_indicator
        gtc_metric["is_filter"] = is_filter
        gtc_metric["ignore_space"] = ignore_space
        gtc_metric["stream"] = stream
        gtc_metric["with_ratio"] = with_ratio
        gtc_metric["max_len"] = max_len
        gtc_metric["max_ratio"] = max_ratio

        self.gtc_metric = build_metric(gtc_metric)
        self.ctc_metric = build_metric(ctc_metric)
        self.latest_sample_ned: Optional[torch.Tensor] = None

    def __call__(self, pred_label, batch=None, training=False, *args, **kwargs):
        ctc_metric = self.ctc_metric(pred_label[1], batch, training=training)
        gtc_metric = self.gtc_metric(pred_label[0], batch, training=training)
        ctc_metric["gtc_acc"] = gtc_metric["acc"]
        ctc_metric["gtc_norm_edit_dis"] = gtc_metric["norm_edit_dis"]

        sample_ned = self.get_last_sample_ned()
        if sample_ned is not None:
            ctc_metric["sample_ned"] = sample_ned

        return ctc_metric

    def get_last_sample_ned(self) -> Optional[torch.Tensor]:
        """Return sample-wise NED from inner ctc_metric if available."""
        getter = getattr(self.ctc_metric, "get_last_sample_ned", None)
        if callable(getter):
            sample_ned = getter()
            if sample_ned is None:
                self.latest_sample_ned = None
            elif torch.is_tensor(sample_ned):
                self.latest_sample_ned = sample_ned.detach().clone()
            else:
                self.latest_sample_ned = torch.as_tensor(
                    sample_ned, dtype=torch.float32)
            return self.latest_sample_ned
        return self.latest_sample_ned

    def get_metric(self, training=False):
        ctc_metric = self.ctc_metric.get_metric(training=training)
        gtc_metric = self.gtc_metric.get_metric(training=training)
        ctc_metric["gtc_acc"] = gtc_metric["acc"]
        ctc_metric["gtc_norm_edit_dis"] = gtc_metric["norm_edit_dis"]
        return ctc_metric
