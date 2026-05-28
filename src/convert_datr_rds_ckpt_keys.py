"""One-off checkpoint key converter for DATR RDS rename.

Converts:
  - reading_direction_selector.horizontal_head.* -> reading_direction_selector.direction_heads.right.*
  - reading_direction_selector.vertical_head.*   -> reading_direction_selector.direction_heads.down.*

Edit INPUT_CKPT_PATH / OUTPUT_CKPT_PATH in main() before running.
"""

from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch


def _rename_key(key: str) -> str:
    key = key.replace(
        "reading_direction_selector.horizontal_head.",
        "reading_direction_selector.direction_heads.right.",
    )
    key = key.replace(
        "reading_direction_selector.vertical_head.",
        "reading_direction_selector.direction_heads.down.",
    )
    return key


def _convert_state_dict(state_dict: Dict[str, Any]) -> Tuple[OrderedDict, int]:
    converted = OrderedDict()
    changed = 0
    for k, v in state_dict.items():
        nk = _rename_key(k)
        if nk != k:
            changed += 1
        converted[nk] = v
    return converted, changed


def main():
    # TODO: edit these two paths before running.
    INPUT_CKPT_PATH = "/home/resources/gch/4_work/4_AIHub_v1.1___hv_aware_log_scale/4_all_x2___synth_hv/1_origin_x2___synth_hv/C/21_datr_v1.1/svtr2_smtr_gtc_rctc___log_2___DPE_sample_adp_1th___from_e4_of_329___id_339/train___id_1/weights/latest.pth"
    OUTPUT_CKPT_PATH = "/home/resources/gch/4_work/4_AIHub_v1.1___hv_aware_log_scale/4_all_x2___synth_hv/1_origin_x2___synth_hv/C/21_datr_v1.1/svtr2_smtr_gtc_rctc___log_2___DPE_sample_adp_1th___from_e4_of_329___id_339/train___id_1/weights/latest2.pth"

    checkpoint = torch.load(INPUT_CKPT_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        original_state_dict = checkpoint["state_dict"]
        converted_state_dict, changed = _convert_state_dict(original_state_dict)
        checkpoint["state_dict"] = converted_state_dict
        save_obj = checkpoint
    elif isinstance(checkpoint, dict):
        converted_state_dict, changed = _convert_state_dict(checkpoint)
        save_obj = converted_state_dict
    else:
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}. "
            "Expected dict with 'state_dict' or raw state_dict dict."
        )

    torch.save(save_obj, OUTPUT_CKPT_PATH)
    print(f"[done] converted keys: {changed}")
    print(f"[in ] {INPUT_CKPT_PATH}")
    print(f"[out] {OUTPUT_CKPT_PATH}")


if __name__ == "__main__":
    main()
