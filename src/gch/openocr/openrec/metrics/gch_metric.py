from openocr.openrec.metrics import build_metric, MODULES
import torch

MODULES["RecGCHMetric"] = "gch.openocr.openrec.metrics.gch_metric"
MODULES["QualityWrapperMetric"] = "gch.openocr.openrec.metrics.gch_metric"
MODULES["QualityMetric"] = "gch.openocr.openrec.metrics.gch_metric"


def _apply_shared_recmetric_args(
    metric_cfg,
    main_indicator="acc",
    is_filter=False,
    ignore_space=True,
    stream=False,
    with_ratio=False,
    max_len=25,
    max_ratio=4,
):
    metric_cfg["main_indicator"] = main_indicator
    metric_cfg["is_filter"] = is_filter
    metric_cfg["ignore_space"] = ignore_space
    metric_cfg["stream"] = stream
    metric_cfg["with_ratio"] = with_ratio
    metric_cfg["max_len"] = max_len
    metric_cfg["max_ratio"] = max_ratio


class RecGCHMetric(object):
    def __init__(
        self,
        c_metric,
        g_metric,
        e_metric,
        use_c: bool = True,
        use_g: bool = True,
        use_e: bool = False,
        main_indicator="acc",
        is_filter=False,
        ignore_space=True,
        stream=False,
        with_ratio=False,
        max_len=25,
        max_ratio=4,
    ):
        self.use_c = use_c
        self.use_g = use_g
        self.use_e = use_e

        _apply_shared_recmetric_args(
            c_metric,
            main_indicator=main_indicator,
            is_filter=is_filter,
            ignore_space=ignore_space,
            stream=stream,
            with_ratio=with_ratio,
            max_len=max_len,
            max_ratio=max_ratio,
        )
        _apply_shared_recmetric_args(
            g_metric,
            main_indicator=main_indicator,
            is_filter=is_filter,
            ignore_space=ignore_space,
            stream=stream,
            with_ratio=with_ratio,
            max_len=max_len,
            max_ratio=max_ratio,
        )

        self.c_metric = build_metric(c_metric)
        self.g_metric = build_metric(g_metric)
        self.e_metric = build_metric(e_metric)
        self.o_metric = build_metric(e_metric)

    def __call__(self, pred_label, batch=None, training=False):
        assert isinstance(batch, dict), "batch must be a dict in this module"
        result = {}
        if self.use_c:
            result["c_metric"] = self.c_metric(
                pred_label["c_pred"], batch["c_label"], training=training
            )
        if self.use_g:
            result["g_metric"] = self.g_metric(
                pred_label["g_pred"], batch["g_label"], training=training
            )

        if self.use_e:
            result["e_metric"] = self.e_metric(
                pred_label["e_pred"], batch["g_label"], training=training # batch는 내부적으로 사용하지 않으며, 심지어 pred는 g 단위로 맞춰놨기에 batch['g_label'] 을 넘긴다.
            )
            result["o_metric"] = self.o_metric(
                pred_label["o_pred"], batch["g_label"], training=training
            )

        
        return result

    def main_indicator(self) -> str:
        return f"c_metric.{self.c_metric.main_indicator}"

    def get_metric(self):
        result = {}
        if self.use_c:
            for k, v in self.c_metric.get_metric().items():
                result[f"c_metric.{k}"] = v
        if self.use_g:
            for k, v in self.g_metric.get_metric().items():
                result[f"g_metric.{k}"] = v

        if self.use_e:
            for k, v in self.e_metric.get_metric().items():
                result[f"e_metric.{k}"] = v

        return result


class QualityMetric(object):
    def __init__(self, main_indicator="mae", **kwargs):
        self.main_indicator = main_indicator
        self.eps = 1e-5
        self.reset()

    def __call__(self, pred_label, batch=None, training=False):
        pred, target = pred_label
        pred_t = torch.as_tensor(pred, dtype=torch.float32).reshape(-1)
        target_t = torch.as_tensor(
            target, dtype=torch.float32, device=pred_t.device
        ).reshape(-1)
        diff = pred_t - target_t

        mae = torch.mean(torch.abs(diff))
        mse = torch.mean(torch.square(diff))
        rmse = torch.sqrt(mse)

        sample_count = int(diff.numel())
        self.sample_count += sample_count
        self.sum_abs_error += float(torch.sum(torch.abs(diff)))
        self.sum_squared_error += float(torch.sum(torch.square(diff)))

        return {"mae": float(mae), "mse": float(mse), "rmse": float(rmse)}

    def get_metric(self):
        result = {
            "mae": self.sum_abs_error / (self.sample_count + self.eps),
            "mse": self.sum_squared_error / (self.sample_count + self.eps),
            "rmse": (self.sum_squared_error / (self.sample_count + self.eps)) ** 0.5,
            # "num_samples": self.sample_count,
        }
        self.reset()
        return result

    def reset(self):
        self.sample_count = 0
        self.sum_abs_error = 0.0
        self.sum_squared_error = 0.0


class QualityWrapperMetric(object):
    def __init__(
        self,
        inner_metric,
        quality_metric=None,
        infer_quality: bool = True,
        main_indicator="acc",
        is_filter=False,
        ignore_space=True,
        stream=False,
        with_ratio=False,
        max_len=25,
        max_ratio=4,
    ):
        _apply_shared_recmetric_args(
            inner_metric,
            main_indicator=main_indicator,
            is_filter=is_filter,
            ignore_space=ignore_space,
            stream=stream,
            with_ratio=with_ratio,
            max_len=max_len,
            max_ratio=max_ratio,
        )

        self.inner_metric = build_metric(inner_metric)
        self.infer_quality = infer_quality
        if self.infer_quality:
            quality_metric = quality_metric or {"name": "QualityMetric"}
            self.quality_metric = build_metric(quality_metric)
        else:
            self.quality_metric = None

    def __call__(self, pred_label, batch=None, training=False):
        assert isinstance(batch, dict), "batch must be a dict in this module"
        result = {
            "inner": self.inner_metric(
                pred_label["inner_pred"], batch["inner_label"], training=training
            )
        }
        if self.infer_quality and self.quality_metric is not None:
            result["quality"] = self.quality_metric(
                pred_label["quality_pred"], batch["quality_label"], training=training
            )
        return result

    def get_metric(self):
        result = {}
        for k, v in self.inner_metric.get_metric().items():
            result[f"inner.{k}"] = v
        if self.infer_quality and self.quality_metric is not None:
            for k, v in self.quality_metric.get_metric().items():
                result[f"quality.{k}"] = v
        return result