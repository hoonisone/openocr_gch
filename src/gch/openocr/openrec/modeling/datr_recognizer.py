import importlib
import math
import random
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union, cast

import torch
from torch import nn

from openocr.openrec.modeling import MODULES
from openocr.openrec.modeling.decoders import build_decoder
from openocr.openrec.modeling.encoders import build_encoder
from openocr.openrec.modeling.transforms import build_transform

__all__ = [
    "ReadingDirectionSelector",
    "DirectionalPositionalEncoding",
    "DATR",
]

VALID_DIRECTIONS: FrozenSet[str] = frozenset({"right", "left", "down", "up"})
DEFAULT_ALLOWED_DIRECTIONS: Tuple[str, ...] = ("right", "down")

_DIRECTION_ALIASES: Dict[str, str] = {
    "horizontal": "right",
    "vertical": "down",
    "h": "right",
    "v": "down",
}


def normalize_direction_name(name: str) -> str:
    n = name.lower().strip()
    return _DIRECTION_ALIASES.get(n, n)


def align_feature(x: torch.Tensor, direction: str) -> torch.Tensor:
    if direction == "right":
        return x
    if direction == "left":
        return x.flip(3).contiguous()
    if direction == "down":
        return x.transpose(2, 3).contiguous()
    if direction == "up":
        return x.transpose(2, 3).flip(3).contiguous()
    raise ValueError(f"Unknown direction: {direction}")


class ReadingDirectionSelector(nn.Module):
    """Predicts per-direction reading quality scores from feature maps.

    Supports an arbitrary set of allowed directions (right, left, down, up).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        allowed_directions: Optional[List[str]] = None,
        train_time_selecting_strategy: str = "batch_oracle",
        infer_time_selecting_strategy: str = "batch_oracle",
        head_type: str = "mlp",
        share_heads: bool = False,
        pool_type: str = "avg",
        drop_rate: float = 0.0,
        detach_score_input: bool = False,
        horizontal_head_type: Optional[str] = None,
        vertical_head_type: Optional[str] = None,
        share_hv_head: Optional[bool] = None,
        stop_grad_to_encoder: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.pool_type = str(pool_type).lower()
        self.drop_rate = float(drop_rate)
        if stop_grad_to_encoder is not None:
            detach_score_input = stop_grad_to_encoder
        self.detach_score_input = bool(detach_score_input)

        if allowed_directions is None:
            allowed_directions = list(DEFAULT_ALLOWED_DIRECTIONS)
        self.allowed_directions = [normalize_direction_name(d) for d in allowed_directions]
        for d in self.allowed_directions:
            if d not in VALID_DIRECTIONS:
                raise ValueError(
                    f"Invalid direction: {d}. Must be one of {sorted(VALID_DIRECTIONS)}")
        if len(self.allowed_directions) < 1:
            raise ValueError("At least one direction must be allowed.")

        if share_hv_head is not None:
            share_heads = share_hv_head
        self.share_heads = bool(share_heads)

        self.train_time_selecting_strategy = self._normalize_strategy(
            train_time_selecting_strategy)
        self.infer_time_selecting_strategy = self._normalize_strategy(
            infer_time_selecting_strategy)
        self._validate_strategy(self.train_time_selecting_strategy)
        self._validate_strategy(self.infer_time_selecting_strategy)

        base_head_type = str(head_type).lower()
        head_type_overrides: Dict[str, str] = {}
        if horizontal_head_type:
            head_type_overrides["right"] = str(horizontal_head_type).lower()
        if vertical_head_type:
            head_type_overrides["down"] = str(vertical_head_type).lower()

        self.direction_heads = nn.ModuleDict()
        first_head: Optional[nn.Module] = None
        for d in self.allowed_directions:
            if self.share_heads and first_head is not None:
                self.direction_heads[d] = first_head
            else:
                ht = head_type_overrides.get(d, base_head_type)
                head = self._build_head(ht)
                self.direction_heads[d] = head
                if first_head is None:
                    first_head = head

    def _build_head(self, head_type: str) -> nn.Module:
        if head_type == "linear":
            return nn.Linear(self.in_channels, 1)
        if head_type == "mlp":
            return nn.Sequential(
                nn.Linear(self.in_channels, self.hidden_channels),
                nn.GELU(),
                nn.Dropout(self.drop_rate),
                nn.Linear(self.hidden_channels, 1),
            )
        raise ValueError(
            f"Unsupported head_type: {head_type}. Use 'linear' or 'mlp'.")

    def _validate_strategy(self, strategy: str):
        valid = {
            "batch_oracle", "batch_random", "batch_adaptive",
            "batch_score_based",
            "sample_random", "sample_adaptive", "sample_score_based",
        }
        for d in VALID_DIRECTIONS:
            valid.add(f"batch_{d}")
        if strategy not in valid:
            raise ValueError(
                f"Unsupported selecting strategy: {strategy}. "
                f"Use one of {sorted(valid)}.")

    def _normalize_strategy(self, strategy: str) -> str:
        s = str(strategy).lower()
        alias: Dict[str, str] = {
            "oracle": "batch_oracle",
            "random": "batch_random",
            "adaptive": "batch_adaptive",
            "score_based": "sample_score_based",
            "horizontal": "batch_right",
            "vertical": "batch_down",
            "batch_horizontal": "batch_right",
            "batch_vertical": "batch_down",
        }
        return alias.get(s, s)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"ReadingDirectionSelector expects [B, C, H, W], got {x.shape}")
        if self.pool_type == "max":
            return x.amax(dim=(2, 3))
        if self.pool_type == "avg":
            return x.mean(dim=(2, 3))
        if self.pool_type == "avgmax":
            return 0.5 * (x.mean(dim=(2, 3)) + x.amax(dim=(2, 3)))
        raise ValueError(
            f"Unsupported pool_type: {self.pool_type}. "
            "Use 'avg', 'max', or 'avgmax'.")

    def _pick_axis_directions(self, axis: str) -> List[str]:
        if axis == "vertical":
            return [d for d in self.allowed_directions if d in ("down", "up")]
        return [d for d in self.allowed_directions if d in ("right", "left")]

    def _infer_direction_from_shape(self, h: int, w: int) -> str:
        if h > w:
            candidates = self._pick_axis_directions("vertical")
        else:
            candidates = self._pick_axis_directions("horizontal")
        return candidates[0] if candidates else self.allowed_directions[0]

    def _sample_adaptive_direction(self, h: int, w: int) -> str:
        if w <= 0:
            candidates = self._pick_axis_directions("vertical")
            return candidates[0] if candidates else self.allowed_directions[0]
        ratio = float(h) / float(w)
        ratio_sq = ratio * ratio
        p_vertical_axis = ratio_sq / (1.0 + ratio_sq)
        v_dirs = self._pick_axis_directions("vertical")
        h_dirs = self._pick_axis_directions("horizontal")
        if random.random() < p_vertical_axis:
            return random.choice(v_dirs) if v_dirs else random.choice(self.allowed_directions)
        return random.choice(h_dirs) if h_dirs else random.choice(self.allowed_directions)

    def _sample_adaptive_direction_per_sample(self, h: int, w: int,
                                              batch_size: int) -> List[str]:
        return [self._sample_adaptive_direction(h, w) for _ in range(batch_size)]

    def _score_from_feat(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.detach_score_input:
            x = x.detach()
        scores: Dict[str, torch.Tensor] = {}
        for d in self.allowed_directions:
            x_d = align_feature(x, d)
            pooled = self._pool(x_d)
            scores[d] = self.direction_heads[d](pooled).squeeze(-1)
        return scores

    def _choose_direction_by_score(
            self, direction_scores: Dict[str, torch.Tensor]) -> List[str]:
        dir_names = list(direction_scores.keys())
        stacked = torch.stack([direction_scores[d] for d in dir_names], dim=-1)
        best_idx = stacked.argmax(dim=-1)
        return [dir_names[int(idx)] for idx in best_idx.tolist()]

    def _choose_batch_direction_by_score(
            self, direction_scores: Dict[str, torch.Tensor]) -> str:
        means = {d: float(s.mean().item()) for d, s in direction_scores.items()}
        return max(means, key=lambda k: means[k])

    def forward(
        self,
        x: torch.Tensor,
        fallback_h: Optional[int] = None,
        fallback_w: Optional[int] = None,
        need_scores: bool = True,
    ) -> list:
        if x.dim() != 4:
            raise ValueError(
                f"ReadingDirectionSelector expects [B, C, H, W], got {x.shape}")
        bsz = int(x.shape[0])
        h = int(x.shape[-2]) if fallback_h is None else int(fallback_h)
        w = int(x.shape[-1]) if fallback_w is None else int(fallback_w)
        strategy = (self.train_time_selecting_strategy if self.training else
                    self.infer_time_selecting_strategy)

        direction_scores: Optional[Dict[str, torch.Tensor]] = None

        if strategy == "sample_score_based":
            direction_scores = self._score_from_feat(x)
            direction = self._choose_direction_by_score(direction_scores)
        elif strategy == "sample_random":
            direction = random.choices(self.allowed_directions, k=bsz)
            if need_scores:
                direction_scores = self._score_from_feat(x)
        elif strategy == "sample_adaptive":
            direction = self._sample_adaptive_direction_per_sample(h, w, bsz)
            if need_scores:
                direction_scores = self._score_from_feat(x)
        elif strategy == "batch_score_based":
            direction_scores = self._score_from_feat(x)
            direction = self._choose_batch_direction_by_score(direction_scores)
        elif strategy == "batch_random":
            direction = random.choice(self.allowed_directions)
            if need_scores:
                direction_scores = self._score_from_feat(x)
        elif strategy == "batch_adaptive":
            direction = self._sample_adaptive_direction(h, w)
            if need_scores:
                direction_scores = self._score_from_feat(x)
        elif strategy == "batch_oracle":
            direction = self._infer_direction_from_shape(h, w)
            if need_scores:
                direction_scores = self._score_from_feat(x)
        elif strategy.startswith("batch_"):
            forced_dir = normalize_direction_name(strategy[len("batch_"):])
            if forced_dir not in VALID_DIRECTIONS:
                raise ValueError(
                    f"Unknown forced direction in strategy '{strategy}'")
            direction = forced_dir
            if need_scores:
                direction_scores = self._score_from_feat(x)
        else:
            raise ValueError(
                f"Unsupported selecting strategy: {strategy}.")

        return [direction, direction_scores]

    @torch.jit.ignore(drop=False)
    def no_weight_decay(self):
        return {}


class DirectionalPositionalEncoding(nn.Module):

    def __init__(self,
                 dim: int,
                 max_size: int = 256,
                 scale_base: float = 10000.0,
                 feature_type: str = "2d"):
        super().__init__()
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if feature_type not in {"2d", "flatten"}:
            raise ValueError(
                f"feature_type must be '2d' or 'flatten', got {feature_type}")

        self.dim = int(dim)
        self.max_size = int(max_size)
        self.scale_base = float(scale_base)
        self.feature_type = feature_type
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer(
            "pe_1d",
            self._build_sinusoidal_pe_1d(dim=self.dim, max_size=self.max_size),
            persistent=False,
        )  # [1, C, L]

    def _build_sinusoidal_pe_1d(self, dim: int, max_size: int) -> torch.Tensor:
        position = torch.arange(max_size, dtype=torch.float32).unsqueeze(1)
        pe = torch.zeros(max_size, dim, dtype=torch.float32)
        even_dim = (dim + 1) // 2
        div_term = torch.exp(
            torch.arange(0, even_dim, dtype=torch.float32) *
            (-math.log(self.scale_base) / float(dim)))
        pe[:, 0::2] = torch.sin(position * div_term[:pe[:, 0::2].shape[1]])
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        return pe.transpose(0, 1).unsqueeze(0)  # [1, C, L]

    def _get_pe_for_direction(self, direction: str, h: int,
                              w: int) -> torch.Tensor:
        pe_1d = cast(torch.Tensor, self.pe_1d)
        d = normalize_direction_name(direction)
        if d == "right":
            return pe_1d[:, :, :w].unsqueeze(2)  # [1, C, 1, W]
        if d == "left":
            return pe_1d[:, :, :w].flip(2).unsqueeze(2)
        if d == "down":
            return pe_1d[:, :, :h].unsqueeze(3)  # [1, C, H, 1]
        if d == "up":
            return pe_1d[:, :, :h].flip(2).unsqueeze(3)
        raise ValueError(f"Unknown direction: {direction}")

    def _apply_pe_2d(self, x: torch.Tensor,
                     direction: Optional[Union[str, List[str]]]) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"DirectionalPositionalEncoding expects [B, C, H, W], got {x.shape}")
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f"dim mismatch: module dim={self.dim}, input C={c}")
        if h > self.max_size or w > self.max_size:
            raise ValueError(
                f"feature size exceeds max_size={self.max_size}: H={h}, W={w}")

        if direction is None or isinstance(direction, str):
            d = normalize_direction_name(direction or "right")
            pe = self._get_pe_for_direction(d, h, w)
            return x + self.pe_scale * pe

        if len(direction) != b:
            raise ValueError(
                f"Direction list length mismatch: len(direction)={len(direction)}, "
                f"batch={b}")

        unique_dirs = set(direction)
        pe_cache: Dict[str, torch.Tensor] = {}
        for d in unique_dirs:
            pe_cache[d] = self._get_pe_for_direction(d, h, w).expand(1, c, h, w)

        pe_mixed = x.new_zeros(b, c, h, w)
        for d in unique_dirs:
            mask = torch.tensor(
                [di == d for di in direction],
                device=x.device, dtype=x.dtype).view(b, 1, 1, 1)
            pe_mixed = pe_mixed + mask * pe_cache[d]
        return x + self.pe_scale * pe_mixed

    def forward(self,
                x: torch.Tensor,
                direction: Optional[Union[str, List[str]]] = None,
                sz: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if self.feature_type == "2d":
            return self._apply_pe_2d(x, direction=direction)

        if x.dim() != 3:
            raise ValueError(
                f"FlattenDirectionalPE expects [B, N, C], got {x.shape}")
        if sz is None or len(sz) != 2:
            raise ValueError("FlattenDirectionalPE requires sz=[H, W].")

        b, n, c = x.shape
        h, w = int(sz[0]), int(sz[1])
        if n != h * w:
            raise ValueError(
                f"FlattenDirectionalPE shape mismatch: N={n}, H*W={h*w}, sz={sz}")
        if c != self.dim:
            raise ValueError(f"dim mismatch: module dim={self.dim}, input C={c}")

        x_2d = x.transpose(1, 2).reshape(b, c, h, w)
        x_2d = self._apply_pe_2d(x_2d, direction=direction)
        return x_2d.flatten(2).transpose(1, 2).contiguous()

    @torch.jit.ignore(drop=False)
    def no_weight_decay(self):
        return {}


class DATR(nn.Module):

    def __init__(self, config: Dict[str, Any]):
        super(DATR, self).__init__()
        in_channels = config.get("in_channels", 3)
        self.use_wd = config.get("use_wd", True)

        self.return_selector_outputs = bool(
            config.get("return_selector_outputs", False))

        if "Transform" not in config or config["Transform"] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config["Transform"]["in_channels"] = in_channels
            self.transform = build_transform(config["Transform"])
            in_channels = self.transform.out_channels

        if "PreEncoder" not in config or config["PreEncoder"] is None:
            self.use_pre_encoder = False
        else:
            self.use_pre_encoder = True
            config["PreEncoder"]["in_channels"] = in_channels
            self.pre_encoder = build_encoder(config["PreEncoder"])
            in_channels = self.pre_encoder.out_channels

        selector_cfg = config.get("ReadingDirectionSelector", None)
        if selector_cfg is None:
            selector_cfg = {"builder": "builtin"}
        else:
            selector_cfg = dict(selector_cfg)
        if "train_time_selecting_strategy" not in selector_cfg and (
                "train_time_selecting_strategy" in config):
            selector_cfg["train_time_selecting_strategy"] = config[
                "train_time_selecting_strategy"]
        if "infer_time_selecting_strategy" not in selector_cfg:
            if "infer_time_selecting_strategy" in config:
                selector_cfg["infer_time_selecting_strategy"] = config[
                    "infer_time_selecting_strategy"]
            elif "infer_fime_selecting_strategy" in config:
                selector_cfg["infer_time_selecting_strategy"] = config[
                    "infer_fime_selecting_strategy"]
        self.reading_direction_selector = self._build_reading_direction_selector(
            selector_cfg, in_channels)

        dpe_cfg = config.get("DirectionalPositionalEncoding", None)
        if dpe_cfg is None:
            self.use_directional_positional_encoding = False
        else:
            self.use_directional_positional_encoding = True
            dpe_cfg = dict(dpe_cfg)
            dpe_cfg["dim"] = dpe_cfg.get("dim", in_channels)
            self.directional_positional_encoding = DirectionalPositionalEncoding(
                **dpe_cfg)

        post_encoder_cfg = config.get("PostEncoder", None)
        if post_encoder_cfg is None and config.get("Encoder", None) is not None:
            post_encoder_cfg = config["Encoder"]

        if post_encoder_cfg is None:
            self.use_post_encoder = False
        else:
            self.use_post_encoder = True
            post_encoder_cfg["in_channels"] = in_channels
            self.post_encoder = build_encoder(post_encoder_cfg)
            in_channels = self.post_encoder.out_channels

        if "Decoder" not in config or config["Decoder"] is None:
            self.use_decoder = False
        else:
            self.use_decoder = True
            config["Decoder"]["in_channels"] = in_channels
            self.decoder = build_decoder(config["Decoder"])

    def _build_reading_direction_selector(self, selector_cfg: Dict[str, Any],
                                          in_channels: int) -> nn.Module:
        cfg = dict(selector_cfg)
        builder = str(cfg.pop("builder", "builtin")).lower()
        cfg["in_channels"] = cfg.get("in_channels", in_channels)

        if builder in {"builtin", "default", "rds"}:
            return ReadingDirectionSelector(**cfg)

        if builder == "encoder":
            return build_encoder(cfg)

        if builder == "module":
            module_path = cfg.pop("module_path", None)
            class_name = cfg.pop("class_name", None)
            if not module_path or not class_name:
                raise ValueError(
                    "ReadingDirectionSelector with builder='module' requires "
                    "'module_path' and 'class_name'.")
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls(**cfg)

        raise ValueError(
            f"Unsupported ReadingDirectionSelector builder: {builder}. "
            "Use 'encoder' or 'module'.")

    def _prepare_feature_by_direction(
            self, x: torch.Tensor, direction: Union[str,
                                                    List[str]]) -> Dict[str, Any]:
        if isinstance(direction, str):
            return {"x": align_feature(x, direction)}
        if len(direction) != x.shape[0]:
            raise ValueError(
                f"Direction list length mismatch: len(direction)={len(direction)}, "
                f"batch={x.shape[0]}")

        dir_indices: Dict[str, List[int]] = {}
        for i, d in enumerate(direction):
            dir_indices.setdefault(d, []).append(i)

        x_by_dir: Dict[str, torch.Tensor] = {}
        idx_by_dir: Dict[str, torch.Tensor] = {}
        for d, indices in dir_indices.items():
            idx_t = torch.tensor(indices, device=x.device, dtype=torch.long)
            x_d = x.index_select(0, idx_t) if idx_t.numel() > 0 else x[:0]
            x_d = align_feature(x_d, d)
            x_by_dir[d] = x_d
            idx_by_dir[d] = idx_t

        return {"x_by_dir": x_by_dir, "idx_by_dir": idx_by_dir}

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

    def _pad_batch_tensor(self, x: torch.Tensor,
                          target_shape: Tuple[int, ...]) -> torch.Tensor:
        if tuple(x.shape[1:]) == target_shape:
            return x
        padded = x.new_zeros((x.shape[0],) + target_shape)
        slices = (slice(None),) + tuple(slice(0, s) for s in x.shape[1:])
        padded[slices] = x
        return padded

    def _merge_multi_outputs(self, pred_by_dir: Dict[str, Any],
                             idx_by_dir: Dict[str, torch.Tensor],
                             total_batch: int) -> Any:
        preds = {d: p for d, p in pred_by_dir.items() if p is not None}
        if len(preds) == 0:
            return None
        if len(preds) == 1:
            return next(iter(preds.values()))

        first = next(iter(preds.values()))

        if isinstance(first, torch.Tensor):
            if first.dim() == 0:
                return sum(preds.values()) / float(len(preds))
            target_shape = list(first.shape[1:])
            for p in preds.values():
                if isinstance(p, torch.Tensor) and p.dim() > 0:
                    for i in range(len(target_shape)):
                        target_shape[i] = max(target_shape[i], p.shape[i + 1])
            ts = tuple(target_shape)
            merged = first.new_zeros((total_batch,) + ts)
            for d, p in preds.items():
                idx = idx_by_dir[d]
                if idx.numel() > 0 and isinstance(p, torch.Tensor):
                    merged.index_copy_(0, idx, self._pad_batch_tensor(p, ts))
            return merged

        if isinstance(first, dict):
            all_keys: set = set()
            for p in preds.values():
                if isinstance(p, dict):
                    all_keys |= p.keys()
            merged_dict: Dict[str, Any] = {}
            for key in all_keys:
                sub = {d: p.get(key) for d, p in preds.items() if isinstance(p, dict)}
                merged_dict[key] = self._merge_multi_outputs(sub, idx_by_dir, total_batch)
            return merged_dict

        if isinstance(first, (list, tuple)):
            lengths = [len(p) for p in preds.values() if isinstance(p, (list, tuple))]
            if len(set(lengths)) == 1:
                merged_items = []
                for i in range(lengths[0]):
                    sub = {d: p[i] for d, p in preds.items() if isinstance(p, (list, tuple))}
                    merged_items.append(self._merge_multi_outputs(sub, idx_by_dir, total_batch))
                return type(first)(merged_items)
            return first

        return first

    def _decode_with_direction(self, feature_pack: Dict[str, Any], data: Any,
                               direction: Union[str, List[str]]) -> Dict[str, Any]:
        if "x" in feature_pack:
            pred = self.decoder(feature_pack["x"], data=data, direction=direction)
            return {
                "pred": pred,
                "pred_by_dir": None,
                "idx_by_dir": None,
            }

        x_by_dir = feature_pack["x_by_dir"]
        idx_by_dir = feature_pack["idx_by_dir"]
        total_batch = sum(idx.numel() for idx in idx_by_dir.values())

        pred_by_dir: Dict[str, Any] = {}
        for d in x_by_dir:
            idx = idx_by_dir[d]
            if idx.numel() > 0:
                data_d = self._split_data_by_index(data, idx, total_batch)
                pred_by_dir[d] = self.decoder(x_by_dir[d], data=data_d)
            else:
                pred_by_dir[d] = None

        return {
            "pred": None,
            "pred_by_dir": pred_by_dir,
            "idx_by_dir": idx_by_dir,
        }

    @torch.jit.ignore(drop=False)
    def no_weight_decay(self):
        if not self.use_wd:
            return {}

        no_weight_decay: Dict[str, Any] = {}
        if self.use_pre_encoder:
            fn = getattr(self.pre_encoder, "no_weight_decay", None)
            if callable(fn):
                ret = fn()
                if isinstance(ret, dict):
                    no_weight_decay.update(ret)
        fn = getattr(self.reading_direction_selector, "no_weight_decay", None)
        if callable(fn):
            ret = fn()
            if isinstance(ret, dict):
                no_weight_decay.update(ret)
        if self.use_directional_positional_encoding:
            fn = getattr(self.directional_positional_encoding, "no_weight_decay",
                         None)
            if callable(fn):
                ret = fn()
                if isinstance(ret, dict):
                    no_weight_decay.update(ret)
        if self.use_post_encoder:
            fn = getattr(self.post_encoder, "no_weight_decay", None)
            if callable(fn):
                ret = fn()
                if isinstance(ret, dict):
                    no_weight_decay.update(ret)
        if self.use_decoder:
            fn = getattr(self.decoder, "no_weight_decay", None)
            if callable(fn):
                ret = fn()
                if isinstance(ret, dict):
                    no_weight_decay.update(ret)
        return no_weight_decay

    def _backward_compat_output(self, result: Dict[str, Any],
                                direction_scores: Optional[Dict[str, torch.Tensor]]) -> None:
        if direction_scores is not None:
            result["h_score"] = direction_scores.get("right")
            result["v_score"] = direction_scores.get("down")
        else:
            result["h_score"] = None
            result["v_score"] = None

        pred_by_dir = result.get("pred_by_dir")
        idx_by_dir = result.get("idx_by_dir")
        if isinstance(pred_by_dir, dict):
            result["pred_h"] = pred_by_dir.get("right")
            result["pred_v"] = pred_by_dir.get("down")
        else:
            result["pred_h"] = None
            result["pred_v"] = None
        if isinstance(idx_by_dir, dict):
            result["idx_h"] = idx_by_dir.get("right")
            result["idx_v"] = idx_by_dir.get("down")
        else:
            result["idx_h"] = None
            result["idx_v"] = None

    def forward(self, x, data=None):
        if x.dim() < 4:
            raise ValueError(f"Expected input with shape [B, C, H, W], got {x.shape}")
        in_h, in_w = int(x.shape[-2]), int(x.shape[-1])

        if self.use_transform:
            x = self.transform(x)

        if self.use_pre_encoder:
            x = self.pre_encoder(x)

        direction, direction_scores = self.reading_direction_selector(
            x,
            fallback_h=in_h,
            fallback_w=in_w,
            need_scores=self.return_selector_outputs,
        )

        if self.use_directional_positional_encoding:
            x = self.directional_positional_encoding(x, direction=direction)

        if self.use_post_encoder:
            x = self.post_encoder(x)

        feature_pack = self._prepare_feature_by_direction(x, direction)

        if self.use_decoder:
            result = self._decode_with_direction(feature_pack, data, direction)
        else:
            if "x" in feature_pack:
                result = {
                    "pred": feature_pack["x"],
                    "pred_by_dir": None,
                    "idx_by_dir": None,
                }
            else:
                result = {
                    "pred": None,
                    "pred_by_dir": feature_pack["x_by_dir"],
                    "idx_by_dir": feature_pack["idx_by_dir"],
                }

        result["selected_direction"] = direction
        result["direction_scores"] = direction_scores
        self._backward_compat_output(result, direction_scores)
        return result


MODULES[
    "DATR"
] = "gch.openocr.openrec.modeling.datr_recognizer"
