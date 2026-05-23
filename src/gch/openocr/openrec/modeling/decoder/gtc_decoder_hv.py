from openocr.openrec.modeling.decoders import build_decoder, class_to_module
from torch import nn
import torch

class_to_module['GTCDecoder_HV'] = (
    'gch.openocr.openrec.modeling.decoder.gtc_decoder_hv')

class GTCDecoder_HV(nn.Module):
    def __init__(
        self,
        in_channels,
        gtc_decoder,
        ctc_decoder,
        detach=False,
        infer_gtc=False,
        out_channels=0,
        rotation_mode="transpose",
        gate_type="token_slot",
        apply_to_gtc:bool = True,
        **kwargs,
    ):
        super().__init__()
        self.detach = detach
        self.infer_gtc = infer_gtc
        self.rotation_mode = rotation_mode
        self.apply_to_gtc = apply_to_gtc

        if infer_gtc:
            gtc_decoder["out_channels"] = out_channels[0]
            ctc_decoder["out_channels"] = out_channels[1]
            gtc_decoder["in_channels"] = in_channels
            ctc_decoder["in_channels"] = in_channels
            self.gtc_decoder = build_decoder(gtc_decoder)
        else:
            ctc_decoder["in_channels"] = in_channels
            ctc_decoder["out_channels"] = out_channels

        self.ctc_decoder = build_decoder(ctc_decoder)

        if rotation_mode == "fusion":
            self.fusion_module = HVTokenSlotFusion_BCHW(
                dim=in_channels,
                num_slots=4,
                num_heads=8,
                gate_type=gate_type,
            )

    def _to_horizontal_feature(self, x):
        if self.rotation_mode == "transpose":
            return x.transpose(2, 3).contiguous()

        if self.rotation_mode == "rot90_ccw":
            return torch.rot90(x, k=1, dims=(2, 3)).contiguous()

        if self.rotation_mode == "rot90_cw":
            return torch.rot90(x, k=-1, dims=(2, 3)).contiguous()

        if self.rotation_mode == "fusion":
            return self.fusion_module(x)

        raise ValueError(f"Unknown rotation_mode: {self.rotation_mode}")

    def forward(self, x, data=None):
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape

        if self.rotation_mode == "fusion":
            # fused: [B, C, L, M], L=max(H,W)
            fused = self.fusion_module(x)

            # CTC/RCTC는 마지막 축을 time W로 보기 때문에
            # [B, C, L, M] 그대로 넣으면 time length = M이 됨.
            # 따라서 [B, C, M, L]로 바꿔서 time length = L이 되게 한다.
            x_ctc = fused.permute(0, 1, 3, 2).contiguous()  # [B, C, M, L]

            # SMTR/GTC는 [B, N, C] visual sequence를 받게 한다.
            if self.apply_to_gtc:
                x_gtc = fused.flatten(2).transpose(1, 2).contiguous()  # [B, L*M, C]
            else:
                x_gtc = x.flatten(2).transpose(1, 2).contiguous()

        else:
            # 세로일 때만 눕히고 싶다면 조건을 이렇게 두는 편이 자연스럽다.
            # 필요하면 threshold 조정.
            origin_x = x
            if h > w:
                x = self._to_horizontal_feature(x)

            x_ctc = x
            if self.apply_to_gtc:
                x_gtc = x.flatten(2).transpose(1, 2).contiguous()
            else:
                x_gtc = origin_x.flatten(2).transpose(1, 2).contiguous()

        ctc_input = x_ctc.detach() if self.detach else x_ctc
        ctc_pred = self.ctc_decoder(ctc_input, data=data)

        if self.training or self.infer_gtc:
            gtc_pred = self.gtc_decoder(x_gtc, data=data)
            return {"gtc_pred": gtc_pred, "ctc_pred": ctc_pred}

        return ctc_pred

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchQueryResampler(nn.Module):
    """
    Variable-length patch group [B_group, P, C]를
    fixed M slots [B_group, M, C]로 resample.

    여기서 P는 horizontal token이면 H,
    vertical token이면 W.
    """

    def __init__(
        self,
        dim: int,
        num_slots: int = 4,   # M
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots

        self.queries = nn.Parameter(torch.zeros(1, num_slots, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: [B_group, P, C]
        return:  [B_group, M, C]
        """
        b_group, _, c = patches.shape

        q = self.queries.expand(b_group, -1, -1)  # [B_group, M, C]

        out, _ = self.attn(
            query=self.q_norm(q),
            key=self.kv_norm(patches),
            value=self.kv_norm(patches),
            need_weights=False,
        )  # [B_group, M, C]

        out = self.out_norm(out)
        return out
class HVTokenSlotFusion_BCHW(nn.Module):
    """
    입력:
        x: [B, C, H, W]

    가로 방향:
        W개 token, 각 token은 H개 patch
        H개 patch -> M slots
        결과: [B, C, W, M]

    세로 방향:
        H개 token, 각 token은 W개 patch
        W개 patch -> M slots
        결과: [B, C, H, M]

    token 개수 정렬:
        둘 다 [B, C, max(H,W), M]

    fusion:
        weighted sum
        결과: [B, C, max(H,W), M]
    """

    def __init__(
        self,
        dim: int,
        num_slots: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        share_resampler: bool = False,
        gate_type: str = "token_slot",  # "token_slot", "token", "global"
    ):
        super().__init__()

        if gate_type not in {"token_slot", "token", "global"}:
            raise ValueError(f"Unknown gate_type: {gate_type}")

        self.dim = dim
        self.num_slots = num_slots
        self.gate_type = gate_type

        self.h_resampler = PatchQueryResampler(
            dim=dim,
            num_slots=num_slots,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.v_resampler = (
            self.h_resampler
            if share_resampler
            else PatchQueryResampler(
                dim=dim,
                num_slots=num_slots,
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )

        self.out_norm = nn.GroupNorm(1, dim)

    @staticmethod
    def resize_token_axis(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        x: [B, C, L, M]
        token axis L만 target_len으로 linear interpolation.
        M slot axis는 유지.

        return: [B, C, target_len, M]
        """
        b, c, l, m = x.shape
        if l == target_len:
            return x

        # linear interpolate는 [N, C, L] 입력을 기대하므로
        # M slot을 batch 쪽으로 접는다.
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, M, C, L]
        x = x.reshape(b * m, c, l)              # [B*M, C, L]

        x = F.interpolate(
            x,
            size=target_len,
            mode="linear",
            align_corners=False,
        )  # [B*M, C, target_len]

        x = x.reshape(b, m, c, target_len)      # [B, M, C, target_len]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, target_len, M]
        return x

    def make_horizontal_slots(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]

        width index마다 하나의 token.
        각 token 안에는 H개 patch가 있음.

        return:
            [B, C, W, M]
        """
        b, c, h, w = x.shape

        # [B, C, H, W] -> [B, W, H, C]
        patches = x.permute(0, 3, 2, 1).contiguous()

        # [B, W, H, C] -> [B*W, H, C]
        patches = patches.reshape(b * w, h, c)

        slots = self.h_resampler(patches)       # [B*W, M, C]

        # [B*W, M, C] -> [B, W, M, C] -> [B, C, W, M]
        slots = slots.reshape(b, w, self.num_slots, c)
        slots = slots.permute(0, 3, 1, 2).contiguous()

        return slots

    def make_vertical_slots(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]

        height index마다 하나의 token.
        각 token 안에는 W개 patch가 있음.

        return:
            [B, C, H, M]
        """
        b, c, h, w = x.shape

        # [B, C, H, W] -> [B, H, W, C]
        patches = x.permute(0, 2, 3, 1).contiguous()

        # [B, H, W, C] -> [B*H, W, C]
        patches = patches.reshape(b * h, w, c)

        slots = self.v_resampler(patches)       # [B*H, M, C]

        # [B*H, M, C] -> [B, H, M, C] -> [B, C, H, M]
        slots = slots.reshape(b, h, self.num_slots, c)
        slots = slots.permute(0, 3, 1, 2).contiguous()

        return slots

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        x: [B, C, H, W]

        return:
            fused: [B, C, max(H,W), M]
        """
        if x.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {x.shape}")

        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f"dim mismatch: module dim={self.dim}, input C={c}")

        h_slots = self.make_horizontal_slots(x)  # [B, C, W, M]
        v_slots = self.make_vertical_slots(x)    # [B, C, H, M]

        target_len = max(h, w)

        h_aligned = self.resize_token_axis(h_slots, target_len)
        v_aligned = self.resize_token_axis(v_slots, target_len)

        if self.gate_type == "token_slot":
            # [B, C, L, M] -> [B, L, M, C]
            h_g = h_aligned.permute(0, 2, 3, 1).contiguous()
            v_g = v_aligned.permute(0, 2, 3, 1).contiguous()

            gate_logits = self.gate(torch.cat([h_g, v_g], dim=-1))
            gate = F.softmax(gate_logits, dim=-1)  # [B, L, M, 2]

            gate = gate.permute(0, 3, 1, 2).contiguous()  # [B, 2, L, M]

            fused = (
                gate[:, 0:1] * h_aligned
                + gate[:, 1:2] * v_aligned
            )

        elif self.gate_type == "token":
            # M slots 평균으로 token-level gate 계산
            h_pool = h_aligned.mean(dim=3).transpose(1, 2)  # [B, L, C]
            v_pool = v_aligned.mean(dim=3).transpose(1, 2)  # [B, L, C]

            gate_logits = self.gate(torch.cat([h_pool, v_pool], dim=-1))
            gate = F.softmax(gate_logits, dim=-1)  # [B, L, 2]

            gate = gate.permute(0, 2, 1).unsqueeze(-1)  # [B, 2, L, 1]

            fused = (
                gate[:, 0:1] * h_aligned
                + gate[:, 1:2] * v_aligned
            )

        else:
            # sample마다 h/v gate 하나
            h_pool = h_aligned.mean(dim=(2, 3))  # [B, C]
            v_pool = v_aligned.mean(dim=(2, 3))  # [B, C]

            gate_logits = self.gate(torch.cat([h_pool, v_pool], dim=-1))
            gate = F.softmax(gate_logits, dim=-1)  # [B, 2]

            fused = (
                gate[:, 0:1, None, None] * h_aligned
                + gate[:, 1:2, None, None] * v_aligned
            )

        fused = self.out_norm(fused)

        if return_aux:
            return {
                "fused": fused,
                "horizontal_slots": h_slots,
                "vertical_slots": v_slots,
                "horizontal_aligned": h_aligned,
                "vertical_aligned": v_aligned,
                "gate": gate,
            }

        return fused