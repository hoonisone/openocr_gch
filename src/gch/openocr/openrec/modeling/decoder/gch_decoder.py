from openocr.openrec.modeling.decoders import build_decoder, class_to_module
from torch import nn
from typing import Optional
import torch


class GCHDecoder(nn.Module):
    def __init__(self, c_decoder, g_decoder, in_channels, 
        use_c:bool = True, 
        use_g:bool = True, 
        # use_distance_head:bool = True,
        **kwargs
    ):
        super(GCHDecoder, self).__init__()
        c_decoder['in_channels'] = in_channels
        g_decoder['in_channels'] = in_channels
        self.c_decoder = build_decoder(c_decoder)
        self.g_decoder = build_decoder(g_decoder)
        self.use_c = use_c
        self.use_g = use_g
        # self.use_distance_head = use_distance_head
        # self.c_distance_head = DistanceHead(in_channels=self.c_decoder.out_channels, min_channels = 512, out_channels=1) if self.use_distance_head else None
        # self.g_distance_head = DistanceHead(in_channels=self.g_decoder.out_channels, min_channels = 512, out_channels=1) if self.use_distance_head else None

    def forward(self, x, data=None):
        
        assert isinstance(data, dict), f"In this module 'GCHDecoder', Arg 'data' must be a dict"
        result = {}
        if self.use_c:
            result['c_pred'] = self.c_decoder(x, data['c_label'])
        #     if self.use_distance_head:
        #         feats, result['c_pred'] = self.c_decoder(x, data['c_label'], return_feats=True)
        #         result['c_dist_pred'] = self.c_distance_head(feats)
        # else:

        if self.use_g:
            result['g_pred'] = self.g_decoder(x, data['g_label'])
            # if self.use_distance_head:
            #     feats, result['g_pred'] = self.g_decoder(x, data['g_label'], return_feats=True)
            #     result['g_dist_pred'] = self.g_distance_head(feats)
            # else:
        return result

class QualityWrapperDecoder(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        max_len,
        inner_decoder,
        infer_quality:bool = True,
        **kwargs
    ):
        super(QualityWrapperDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        inner_decoder['in_channels'] = in_channels
        inner_decoder['out_channels'] = out_channels
        self.inner_decoder = build_decoder(inner_decoder)
        self.infer_distance = infer_quality

        self.quality_head = DistanceHead(
            in_channels=self.inner_decoder.feature_channels, 
            max_len=max_len
        )

        # self.distance_head = DistanceHead(in_channels=self.decoder.feature_channels, out_channels=1)

    def forward(self, x, data=None):
        if self.infer_distance:
            feats, pred = self.inner_decoder(x, data=data["inner_label"], return_feats=True)
            distance = self.quality_head(pred, data["quality_label"])
            return {"inner_pred": pred, "quality_pred": distance}
        else:
            return {"inner_pred": self.inner_decoder(x, data=data["inner_label"])}


import torch
import torch.nn.functional as F


def extract_ctc_quality_features(
    x: torch.Tensor,
    blank_id: int = 0,
    input_lengths: torch.Tensor | None = None,
    is_log_probs: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Extract sequence-level quality features from CTC outputs.

    Args:
        x:
            Tensor of shape [B, T, C].
            - If is_log_probs=False: raw logits
            - If is_log_probs=True: log-probabilities
        blank_id:
            Index of the CTC blank token.
        input_lengths:
            Optional tensor of shape [B] with valid time lengths.
            Frames beyond each length are ignored.
        is_log_probs:
            Whether x is already log_softmax-ed.
        eps:
            Small constant for numerical stability.

    Returns:
        features:
            Tensor of shape [B, 9], in the following order:
            [
                mean_maxprob,
                min_maxprob,
                mean_entropy,
                blank_ratio,
                repetition_ratio,
                decoded_length,
                logprob_per_char,
                margin_mean,
                margin_min,
            ]
    """
    if x.dim() != 3:
        raise ValueError(f"Expected x to have shape [B, T, C], got {tuple(x.shape)}")

    B, T, C = x.shape
    device = x.device

    if input_lengths is None:
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    else:
        input_lengths = input_lengths.to(device=device, dtype=torch.long)
        if input_lengths.shape != (B,):
            raise ValueError(
                f"Expected input_lengths shape [B], got {tuple(input_lengths.shape)}"
            )
        if torch.any(input_lengths < 1) or torch.any(input_lengths > T):
            raise ValueError("input_lengths must satisfy 1 <= input_lengths <= T")

    # [B, T]
    time_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    valid_mask = time_ids < input_lengths.unsqueeze(1)

    # Convert to log_probs / probs
    if is_log_probs:
        log_probs = x
    else:
        log_probs = F.log_softmax(x, dim=-1)

    probs = log_probs.exp()  # [B, T, C]

    # Top-1 / Top-2 stats per frame
    top2_probs, top2_ids = probs.topk(k=min(2, C), dim=-1)  # [B, T, 2] if C>=2
    max_prob = top2_probs[..., 0]                           # [B, T]
    argmax_id = top2_ids[..., 0]                            # [B, T]

    if C >= 2:
        second_prob = top2_probs[..., 1]
        margin = max_prob - second_prob
    else:
        second_prob = torch.zeros_like(max_prob)
        margin = max_prob  # degenerate case

    # Entropy per frame: -sum p log p
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, T]

    # Masked reductions helper
    valid_mask_f = valid_mask.float()
    valid_count = valid_mask_f.sum(dim=1).clamp_min(1.0)  # [B]

    def masked_mean(v: torch.Tensor) -> torch.Tensor:
        return (v * valid_mask_f).sum(dim=1) / valid_count

    def masked_min(v: torch.Tensor, fill_value: float) -> torch.Tensor:
        filled = torch.where(valid_mask, v, torch.full_like(v, fill_value))
        return filled.min(dim=1).values

    # 1) mean_maxprob
    mean_maxprob = masked_mean(max_prob)

    # 2) min_maxprob
    min_maxprob = masked_min(max_prob, fill_value=float("inf"))

    # 3) mean_entropy
    mean_entropy = masked_mean(entropy)

    # 4) blank_ratio
    is_blank = (argmax_id == blank_id) & valid_mask
    blank_ratio = is_blank.float().sum(dim=1) / valid_count

    # 5) repetition_ratio
    # Count repeated argmax between consecutive valid frames:
    # argmax[t] == argmax[t-1], both valid, and optionally non-blank
    if T > 1:
        curr_valid = valid_mask[:, 1:]
        prev_valid = valid_mask[:, :-1]
        pair_valid = curr_valid & prev_valid

        curr_id = argmax_id[:, 1:]
        prev_id = argmax_id[:, :-1]

        repeated = (curr_id == prev_id) & pair_valid
        # Usually more meaningful to exclude blank repetitions
        repeated = repeated & (curr_id != blank_id)

        pair_count = pair_valid.float().sum(dim=1).clamp_min(1.0)
        repetition_ratio = repeated.float().sum(dim=1) / pair_count
    else:
        repetition_ratio = torch.zeros(B, device=device)

    # 6) decoded_length
    # Greedy CTC decode length = number of non-blank tokens after collapse
    # Keep token if:
    #   - valid
    #   - non-blank
    #   - first valid frame OR token != previous token
    if T > 0:
        first_frame_keep = valid_mask[:, :1] & (argmax_id[:, :1] != blank_id)

        if T > 1:
            changed = argmax_id[:, 1:] != argmax_id[:, :-1]
            keep_rest = valid_mask[:, 1:] & (argmax_id[:, 1:] != blank_id) & changed
            keep_mask = torch.cat([first_frame_keep, keep_rest], dim=1)
        else:
            keep_mask = first_frame_keep

        decoded_length = keep_mask.float().sum(dim=1)
    else:
        decoded_length = torch.zeros(B, device=device)

    # 7) logprob_per_char
    # Best-path log-prob per valid frame, normalized by decoded length
    best_path_logprob = torch.gather(
        log_probs, dim=-1, index=argmax_id.unsqueeze(-1)
    ).squeeze(-1)  # [B, T]
    best_path_logprob_sum = (best_path_logprob * valid_mask_f).sum(dim=1)

    # Normalize by decoded length; if decoded_length == 0, fall back to valid length
    denom = torch.where(decoded_length > 0, decoded_length, valid_count)
    logprob_per_char = best_path_logprob_sum / denom.clamp_min(1.0)

    # 8) margin_mean
    margin_mean = masked_mean(margin)

    # 9) margin_min
    margin_min = masked_min(margin, fill_value=float("inf"))

    features = torch.stack(
        [
            mean_maxprob,
            min_maxprob,
            mean_entropy,
            blank_ratio,
            repetition_ratio,
            decoded_length,
            logprob_per_char,
            margin_mean,
            margin_min,
        ],
        dim=1,
    )  # [B, 9]

    return features

class DistanceHead(nn.Module):
    def __init__(self, in_channels:int,
        max_len:int, 
        min_channels:Optional[int] = None, **kwargs):


        super(DistanceHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 1
        self.fc = nn.Linear(in_channels, self.out_channels)
        self.fc_temp = nn.Linear(9, self.out_channels)

    def forward(self, x, data=None):
        # RCTC: train → logits; eval → softmax probs. Match that here.
        if self.training:
            y = extract_ctc_quality_features(x, is_log_probs=False)
        else:
            log_probs = torch.log(x.clamp_min(1e-8))
            y = extract_ctc_quality_features(log_probs, is_log_probs=True)
        y = self.fc_temp(y)
        
        return torch.sigmoid(y)

class_to_module['GCHDecoder'] = 'gch.openocr.openrec.modeling.decoder.gch_decoder'
class_to_module['QualityWrapperDecoder'] = 'gch.openocr.openrec.modeling.decoder.gch_decoder'