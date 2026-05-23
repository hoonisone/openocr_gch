import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from openocr.openrec.modeling.common import Block
from openocr.openrec.modeling.decoders import class_to_module

class_to_module['RCTCDecoder_HV'] = (
    'gch.openocr.openrec.modeling.decoder.rctc_decoder_hv')

class RCTCDecoder_HV(nn.Module):
    """
    RCTC decoder that automatically chooses horizontal or vertical decoding.

    - forward_h: horizontal text assumption
        input  [B, C, H, W]
        output [B, W, num_classes]

    - forward_v: vertical text assumption
        input  [B, C, H, W]
        output [B, H, num_classes]

    Direction rule:
        if W >= H: use forward_h
        else:      use forward_v
    """

    def __init__(
        self,
        in_channels,
        out_channels=6625,
        return_feats=False,
        mode = "hv_seperate",
        ratio_threshold=0.8,
        vertical_mode = "transpose",
        **kwargs,
    ):
        super().__init__()

        self.ratio_threshold = ratio_threshold
        self.mode = mode
        self.vertical_mode = vertical_mode

        self.char_token_h = nn.Parameter(
            torch.zeros([1, 1, in_channels], dtype=torch.float32),
            requires_grad=True,
        )
        self.char_token_v = nn.Parameter(
            torch.zeros([1, 1, in_channels], dtype=torch.float32),
            requires_grad=True,
        )

        trunc_normal_(self.char_token_h, mean=0, std=0.02)
        trunc_normal_(self.char_token_v, mean=0, std=0.02)

        self.fc = nn.Linear(
            in_channels,
            out_channels,
            bias=True,
        )

        self.fc_kv_h = nn.Linear(
            in_channels,
            2 * in_channels,
            bias=True,
        )
        self.fc_kv_v = nn.Linear(
            in_channels,
            2 * in_channels,
            bias=True,
        )

        num_heads = max(1, in_channels // 32)

        # Width-direction attention block.
        # Used in forward_h over sequences of length W.
        self.w_atten_block = Block(
            dim=in_channels,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
        )

        # Height-direction attention block.
        # Used in forward_v over sequences of length H.
        self.h_atten_block = Block(
            dim=in_channels,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
        )

        self.out_channels = out_channels
        self.return_feats = return_feats
        self.in_channels = in_channels

    @property
    def feature_channels(self) -> int:
        return self.in_channels

    def _post_process(self, feats, return_feats=False):
        """
        feats:
            horizontal: [B, W, C]
            vertical:   [B, H, C]

        returns:
            training: raw logits
            eval:     softmax probabilities
        """
        predicts = self.fc(feats)

        if not self.training:
            predicts = F.softmax(predicts, dim=2)

        if self.return_feats or return_feats:
            return feats, predicts

        return predicts

    def forward_h(self, x, return_feats: bool = False):
        """
        Horizontal RCTC.

        Sequence axis: W
        Aggregation axis: H

        [B, C, H, W]
        -> row-wise width attention
        -> height attention pooling per width position
        -> [B, W, C]
        -> classifier
        """
        B, C, H, W = x.shape

        # Apply self-attention along width for each height row.
        # [B, C, H, W]
        # -> [B, H, W, C]
        # -> [B * H, W, C]
        # -> [B, C, H, W]
        x = self.w_atten_block(
            x.permute(0, 2, 3, 1).reshape(-1, W, C)
        ).reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Build key/value from all 2D positions.
        # x.flatten(2).transpose(1, 2): [B, H * W, C]
        # after fc_kv_h: [B, H * W, 2C]
        # reshape/permute -> [2, B, C, H * W]
        x_kv = self.fc_kv_h(x.flatten(2).transpose(1, 2)).reshape(
            B, H * W, 2, C
        ).permute(2, 0, 3, 1)

        x_k, x_v = x_kv.unbind(0)  # each: [B, C, H * W]

        char_token = self.char_token_h.tile([B, 1, 1])  # [B, 1, C]

        # Attention score over all 2D positions.
        attn_ctc2d = char_token @ x_k  # [B, 1, H * W]
        attn_ctc2d = attn_ctc2d.reshape(B, 1, H, W)

        # For each width position, select/aggregate height positions.
        # softmax over H.
        attn_ctc2d = F.softmax(attn_ctc2d, dim=2)  # [B, 1, H, W]

        # [B, 1, H, W] -> [B, W, 1, H]
        attn_ctc2d = attn_ctc2d.permute(0, 3, 1, 2)

        # Value map.
        x_v = x_v.reshape(B, C, H, W)

        # [B, W, 1, H] @ [B, W, H, C] -> [B, W, 1, C]
        feats = attn_ctc2d @ x_v.permute(0, 3, 2, 1)
        feats = feats.squeeze(2)  # [B, W, C]

        return self._post_process(feats, return_feats=return_feats)

    def forward_v(self, x, return_feats: bool = False):
        """
        Vertical RCTC.

        Sequence axis: H
        Aggregation axis: W

        [B, C, H, W]
        -> column-wise height attention
        -> width attention pooling per height position
        -> [B, H, C]
        -> classifier
        """
        B, C, H, W = x.shape

        # Apply self-attention along height for each width column.
        # [B, C, H, W]
        # -> [B, W, H, C]
        # -> [B * W, H, C]
        # -> [B, C, H, W]
        x = self.h_atten_block(
            x.permute(0, 3, 2, 1).reshape(-1, H, C)
        ).reshape(B, W, H, C).permute(0, 3, 2, 1)

        # Build key/value from all 2D positions.
        # x.flatten(2).transpose(1, 2): [B, H * W, C]
        # after fc_kv_v: [B, H * W, 2C]
        # reshape/permute -> [2, B, C, H * W]
        x_kv = self.fc_kv_v(x.flatten(2).transpose(1, 2)).reshape(
            B, H * W, 2, C
        ).permute(2, 0, 3, 1)

        x_k, x_v = x_kv.unbind(0)  # each: [B, C, H * W]

        char_token = self.char_token_v.tile([B, 1, 1])  # [B, 1, C]

        # Attention score over all 2D positions.
        attn_ctc2d = char_token @ x_k  # [B, 1, H * W]
        attn_ctc2d = attn_ctc2d.reshape(B, 1, H, W)

        # For each height position, select/aggregate width positions.
        # softmax over W.
        attn_ctc2d = F.softmax(attn_ctc2d, dim=3)  # [B, 1, H, W]

        # [B, 1, H, W] -> [B, H, 1, W]
        attn_ctc2d = attn_ctc2d.permute(0, 2, 1, 3)

        # Value map.
        x_v = x_v.reshape(B, C, H, W)

        # [B, H, 1, W] @ [B, H, W, C] -> [B, H, 1, C]
        feats = attn_ctc2d @ x_v.permute(0, 2, 3, 1)
        feats = feats.squeeze(2)  # [B, H, C]

        return self._post_process(feats, return_feats=return_feats)

    def _to_horizontal_feature(self, x):
        """
        Convert vertical feature map to horizontal-reading feature map.

        x: [B, C, H, W]

        vertical_mode:
            "transpose":
                [B, C, H, W] -> [B, C, W, H]
                original top-to-bottom becomes new left-to-right.

            "rot90_ccw":
                torch.rot90(x, k=1, dims=(2, 3))
                [B, C, H, W] -> [B, C, W, H]
                includes flip; sequence order may be reversed depending label convention.

            "rot90_cw":
                torch.rot90(x, k=-1, dims=(2, 3))
                [B, C, H, W] -> [B, C, W, H]
                includes opposite flip.
        """
        if self.vertical_mode == "transpose":
            return x.transpose(2, 3).contiguous()

        if self.vertical_mode == "rot90_ccw":
            return torch.rot90(x, k=1, dims=(2, 3)).contiguous()

        if self.vertical_mode == "rot90_cw":
            return torch.rot90(x, k=-1, dims=(2, 3)).contiguous()

        raise ValueError(f"Unknown vertical_mode: {self.vertical_mode}")

    def forward(self, x, data=None, return_feats: bool = False):
        """
        Automatically choose horizontal or vertical decoding by feature shape.

        W >= H -> horizontal
        H > W  -> vertical
        """
        B, C, H, W = x.shape

        if self.mode == "hv_seperate":
            if W/H > self.ratio_threshold :
                return self.forward_h(x, return_feats=return_feats)
            else:
                return self.forward_v(x, return_feats=return_feats)
        elif self.mode == "v_90_rotate":
            if W / H > self.ratio_threshold:
                return self.forward_h(x, return_feats=return_feats)

            x = self._to_horizontal_feature(x)
            return self.forward_h(x, return_feats=return_feats)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")