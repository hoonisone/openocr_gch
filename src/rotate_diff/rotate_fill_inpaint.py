# rotate_fill_inpaint.py

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def estimate_border_color(img: Image.Image) -> tuple[int, int, int]:
    arr = np.array(img)
    top = arr[0, :, :]
    bottom = arr[-1, :, :]
    left = arr[:, 0, :]
    right = arr[:, -1, :]
    border = np.concatenate([top, bottom, left, right], axis=0)
    color = np.median(border, axis=0).astype(np.uint8)
    return tuple(int(x) for x in color)


def make_rotated_with_empty_mask(
    img: Image.Image,
    angle: float,
    pad_ratio: float = 0.35,
):
    """
    Returns:
        rotated_rgb: 회전 후 빈 공간이 배경색으로 채워진 RGB 이미지
        empty_mask: 빈 공간 영역 mask, PIL L mode, white=fill target
        rotated_debug: 투명 배경 회전 결과 확인용 RGBA
    """
    w, h = img.size
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)

    bg_color = estimate_border_color(img)

    # RGB canvas
    canvas = Image.new("RGB", (w + 2 * pad_x, h + 2 * pad_y), bg_color)
    canvas.paste(img, (pad_x, pad_y))

    # Alpha canvas: 원본이 있는 영역만 255, 나머지는 0
    alpha = Image.new("L", canvas.size, 0)
    alpha.paste(Image.new("L", img.size, 255), (pad_x, pad_y))

    rgba = canvas.convert("RGBA")
    rgba.putalpha(alpha)

    # rotate with transparent empty regions
    rotated_rgba = rgba.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )

    rotated_alpha = rotated_rgba.getchannel("A")

    # empty area: alpha가 낮은 부분
    mask = np.array(rotated_alpha)
    empty = (mask < 250).astype(np.uint8) * 255

    # edge anti-aliasing까지 조금 포함
    kernel = np.ones((5, 5), np.uint8)
    empty = cv2.dilate(empty, kernel, iterations=1)

    empty_mask = Image.fromarray(empty, mode="L")

    # init image는 빈 영역을 border color로 합성
    bg = Image.new("RGB", rotated_rgba.size, bg_color)
    rotated_rgb = Image.alpha_composite(bg.convert("RGBA"), rotated_rgba).convert("RGB")

    return rotated_rgb, empty_mask, rotated_rgba


def opencv_inpaint(rotated_rgb: Image.Image, mask: Image.Image, radius: int = 5) -> Image.Image:
    img_bgr = cv2.cvtColor(np.array(rotated_rgb), cv2.COLOR_RGB2BGR)
    mask_np = np.array(mask)

    result_bgr = cv2.inpaint(
        img_bgr,
        mask_np,
        inpaintRadius=radius,
        flags=cv2.INPAINT_TELEA,
    )

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def resize_to_multiple_of_8(img: Image.Image, mask: Image.Image, max_side: int = 768):
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))

    nw = int(w * scale)
    nh = int(h * scale)

    nw = max(64, (nw // 8) * 8)
    nh = max(64, (nh // 8) * 8)

    img_r = img.resize((nw, nh), Image.Resampling.LANCZOS)
    mask_r = mask.resize((nw, nh), Image.Resampling.NEAREST)

    return img_r, mask_r


def diffusion_inpaint(
    rotated_rgb: Image.Image,
    mask: Image.Image,
    model_id: str,
    output_size_reference: tuple[int, int],
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    max_side: int,
) -> Image.Image:
    import torch
    from diffusers import AutoPipelineForInpainting

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    img_r, mask_r = resize_to_multiple_of_8(rotated_rgb, mask, max_side=max_side)

    generator = torch.Generator(device=device).manual_seed(seed)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img_r,
        mask_image=mask_r,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # 원래 회전 이미지 크기로 복원
    out = out.resize(output_size_reference, Image.Resampling.LANCZOS)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input horizontal text image path")
    parser.add_argument("--out_dir", default="rotation_aug_out")
    parser.add_argument("--angle", type=float, default=15.0)
    parser.add_argument("--pad_ratio", type=float, default=0.35)
    parser.add_argument(
        "--method",
        choices=["opencv", "diffusion", "both"],
        default="opencv",
        help="opencv is fast; diffusion is more natural but slower",
    )

    # diffusion options
    parser.add_argument(
        "--model_id",
        default="stable-diffusion-v1-5/stable-diffusion-inpainting",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--max_side", type=int, default=768)

    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_rgb(str(image_path))

    rotated_rgb, mask, rotated_rgba = make_rotated_with_empty_mask(
        img,
        angle=args.angle,
        pad_ratio=args.pad_ratio,
    )

    stem = image_path.stem

    rotated_rgb.save(out_dir / f"{stem}_rotated_empty.png")
    mask.save(out_dir / f"{stem}_empty_mask.png")
    rotated_rgba.save(out_dir / f"{stem}_rotated_debug_rgba.png")

    if args.method in ["opencv", "both"]:
        cv_result = opencv_inpaint(rotated_rgb, mask)
        cv_result.save(out_dir / f"{stem}_filled_opencv.png")

    if args.method in ["diffusion", "both"]:
        prompt = (
            "natural background texture, clean surface, realistic image, "
            "no text, no letters, no words"
        )
        negative_prompt = (
            "text, letters, words, character, logo, watermark, sign, symbol, "
            "caption, typography, font"
        )

        diff_result = diffusion_inpaint(
            rotated_rgb=rotated_rgb,
            mask=mask,
            model_id=args.model_id,
            output_size_reference=rotated_rgb.size,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            max_side=args.max_side,
        )
        diff_result.save(out_dir / f"{stem}_filled_diffusion.png")

    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()