from PIL import Image
import random
import math
from typing import Union, Tuple, Optional, List, Literal

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


RotateRange = Union[
    float,
    int,
    Tuple[float, float],
    List[float],
    List[List[float]],
    List[Tuple[float, float]],
]
RotateMode = Literal[
    "range_random",
    "aspect_ratio_range_random",
    "aspect_ratio_fixed_random_direction",
]


def sample_angle(
    rotate_range: RotateRange,
    rng: Optional[random.Random] = None,
) -> float:
    """
    rotate_range 형식:
        10
            -> -10도 ~ +10도

        [-15, 15]
            -> -15도 ~ +15도

        [[-20, -10], [10, 20]]
            -> 두 구간 중 하나를 랜덤 선택 후 그 안에서 샘플링
    """

    if rng is None:
        rng = random

    # 숫자 하나인 경우: -abs(x) ~ abs(x)
    if isinstance(rotate_range, (int, float)):
        value = abs(float(rotate_range))
        return rng.uniform(-value, value)

    if not isinstance(rotate_range, (list, tuple)):
        raise ValueError(
            "rotate_range must be a number, [min, max], or [[min, max], ...]"
        )

    # 단일 구간: [-15, 15] 또는 (-15, 15)
    if (
        len(rotate_range) == 2
        and all(isinstance(v, (int, float)) for v in rotate_range)
    ):
        low = float(rotate_range[0])
        high = float(rotate_range[1])

        if low > high:
            raise ValueError(f"Invalid rotate_range: [{low}, {high}]")

        return rng.uniform(low, high)

    # 여러 구간: [[-20, -10], [10, 20]]
    if len(rotate_range) == 0:
        raise ValueError("rotate_range cannot be empty")

    intervals = rotate_range

    for interval in intervals:
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
            or not all(isinstance(v, (int, float)) for v in interval)
        ):
            raise ValueError(
                "Each rotate_range interval must be [min, max], "
                "e.g. [[-20, -10], [10, 20]]"
            )

        low = float(interval[0])
        high = float(interval[1])

        if low > high:
            raise ValueError(f"Invalid rotate interval: [{low}, {high}]")

    selected = rng.choice(intervals)
    return rng.uniform(float(selected[0]), float(selected[1]))


def get_aspect_ratio_angle_limit(width: int, height: int) -> float:
    """
    샘플의 종횡비로부터 atan 절대값 기반 최대 회전 각도를 계산합니다.
    각도 단위는 degree입니다.
    """
    if width <= 0 or height <= 0:
        return 0.0
    base = abs(math.degrees(math.atan(float(height) / float(width))))
    if height > width:
        # 세로가 더 긴 경우에는 90도 기준 여각을 최대 회전 각도로 사용
        return 90.0 - base
    return base


def sample_angle_by_mode(
    rotate_mode: RotateMode,
    rotate_range: RotateRange,
    width: int,
    height: int,
    rng: Optional[random.Random] = None,
) -> float:
    if rng is None:
        rng = random

    if rotate_mode == "range_random":
        return sample_angle(rotate_range, rng=rng)

    aspect_limit = get_aspect_ratio_angle_limit(width=width, height=height)

    if rotate_mode == "aspect_ratio_range_random":
        return rng.uniform(-aspect_limit, aspect_limit)

    if rotate_mode == "aspect_ratio_fixed_random_direction":
        direction = -1.0 if rng.random() < 0.5 else 1.0
        return direction * aspect_limit

    raise ValueError(
        "rotate_mode must be one of: "
        "range_random, aspect_ratio_range_random, "
        "aspect_ratio_fixed_random_direction"
    )


def tile_rotate_augment(
    image: Image.Image,
    use_rotate: bool = True,
    rotate_range: RotateRange = (-10, 10),
    rotate_mode: RotateMode = "range_random",
    rotate_prob: float = 1.0,
    use_tiling: bool = True,
    tiling_prob: float = 1.0,
    crop: bool = True,
    fillcolor: Optional[Tuple[int, int, int]] = None,
    no_tiling_bg_mode: str = "fillcolor",
    rng: Optional[random.Random] = None,
) -> Image.Image:
    """
    이미지를 3x3으로 이어 붙인 뒤 중심 기준으로 회전하고,
    crop=True이면 회전된 중앙 원본 이미지 영역을 모두 포함하는
    최소 axis-aligned bbox로 crop하여 반환합니다.

    Args:
        image:
            이미 로드된 PIL.Image.Image 객체.

        rotate_range:
            회전 각도 범위.

            예:
                10
                    -> -10도 ~ +10도

                [-15, 15]
                    -> -15도 ~ +15도

                [[-20, -10], [10, 20]]
                    -> 여러 구간 중 하나를 랜덤 선택 후 샘플링

        rotate_prob:
            use_rotate=True일 때 회전 적용 확률. 0.0 ~ 1.0

        rotate_mode:
            회전 각도 샘플링 모드.

            - "range_random"
              rotate_range 안에서 랜덤 샘플링 (기존 동작)

            - "aspect_ratio_range_random"
              각 샘플의 atan(h / w) 절대값을 최대 범위로 두고
              [-max, +max]에서 랜덤 샘플링

            - "aspect_ratio_fixed_random_direction"
              각 샘플의 atan(h / w) 절대값으로 고정하고
              시계/반시계 방향만 랜덤 선택

        use_rotate:
            True면 회전 적용을 사용할 수 있음.
            False면 회전을 수행하지 않음.

        use_tiling:
            True면 타일링을 사용할 수 있음.
            False면 타일링 없이 원본 이미지 기준 회전만 수행.

        tiling_prob:
            use_tiling=True일 때 실제로 타일링을 적용할 확률. 0.0 ~ 1.0

        crop:
            True이면 회전된 원본 영역을 모두 포함하는 최소 bbox로 crop.
            False이면 회전된 3x3 타일 이미지를 그대로 반환.

        fillcolor:
            회전 시 바깥 영역 색.

        no_tiling_bg_mode:
            use_tiling=False일 때 배경 채움 방식.
            - "fillcolor": fillcolor(없으면 검정) 사용
            - "mean": 원본 이미지 평균 RGB 사용
            - "inpaint": cv2.inpaint로 채움

    Returns:
        PIL.Image.Image
    """

    if rng is None:
        rng = random

    img = image.convert("RGB")
    w, h = img.size

    use_tiling_now = use_tiling and (rng.random() < tiling_prob)
    if use_rotate and (rng.random() < rotate_prob):
        angle = sample_angle_by_mode(
            rotate_mode=rotate_mode,
            rotate_range=rotate_range,
            width=w,
            height=h,
            rng=rng,
        )
    else:
        angle = 0.0

    def _rotated_bbox(corners, cx, cy, angle_deg):
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rotated_corners = []
        for x, y in corners:
            dx = x - cx
            dy = y - cy
            rx = cx + dx * cos_t - dy * sin_t
            ry = cy + dx * sin_t + dy * cos_t
            rotated_corners.append((rx, ry))
        xs = [p[0] for p in rotated_corners]
        ys = [p[1] for p in rotated_corners]
        left = math.floor(min(xs))
        top = math.floor(min(ys))
        right = math.ceil(max(xs))
        bottom = math.ceil(max(ys))
        return left, top, right, bottom

    def _resolve_bg_color():
        if no_tiling_bg_mode == "mean":
            arr = np.asarray(img, dtype=np.float32)
            mean_rgb = arr.mean(axis=(0, 1))
            return tuple(int(round(v)) for v in mean_rgb.tolist())
        # fillcolor 모드 기본값은 검정
        return fillcolor if fillcolor is not None else (0, 0, 0)

    def _inpaint_background(rotated_img: Image.Image, valid_mask_img: Image.Image):
        if cv2 is None:
            raise ImportError(
                "no_tiling_bg_mode='inpaint' requires opencv-python (cv2)."
            )
        rotated_np = np.array(rotated_img, dtype=np.uint8)
        mask_np = np.array(valid_mask_img, dtype=np.uint8)
        inpaint_mask = (mask_np == 0).astype(np.uint8) * 255
        bgr = cv2.cvtColor(rotated_np, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(inpainted_rgb)

    # 타일링을 사용하지 않는 경우:
    # - crop=False: 원본 캔버스 기준 회전 결과 반환
    # - crop=True : 회전된 원본 4꼭짓점을 포함하는 최소 bbox로 crop
    if not use_tiling_now:
        if angle == 0.0:
            return img
        if not crop:
            return img.rotate(
                angle,
                resample=Image.BICUBIC,
                expand=False,
                center=(w / 2.0, h / 2.0),
                fillcolor=fillcolor,
            )

        canvas_bg = _resolve_bg_color()
        canvas = Image.new("RGB", (w * 3, h * 3), color=canvas_bg)
        canvas.paste(img, (w, h))
        mask_canvas = Image.new("L", (w * 3, h * 3), color=0)
        mask_canvas.paste(Image.new("L", (w, h), color=255), (w, h))
        cx, cy = 1.5 * w, 1.5 * h
        original_corners = [
            (w, h),
            (2 * w, h),
            (2 * w, 2 * h),
            (w, 2 * h),
        ]
        rotated = canvas.rotate(
            angle,
            resample=Image.BICUBIC,
            expand=False,
            center=(cx, cy),
            fillcolor=fillcolor,
        )
        rotated_mask = mask_canvas.rotate(
            angle,
            resample=Image.NEAREST,
            expand=False,
            center=(cx, cy),
            fillcolor=0,
        )
        left, top, right, bottom = _rotated_bbox(original_corners, cx, cy, angle)
        cropped = rotated.crop((left, top, right, bottom))
        if no_tiling_bg_mode == "inpaint":
            cropped_mask = rotated_mask.crop((left, top, right, bottom))
            return _inpaint_background(cropped, cropped_mask)
        return cropped

    # 1. 3x3 타일 생성
    tiled = Image.new("RGB", (w * 3, h * 3))

    for row in range(3):
        for col in range(3):
            tiled.paste(img, (col * w, row * h))

    # 3x3 이미지 중심 == 중앙 원본 이미지 중심
    cx, cy = 1.5 * w, 1.5 * h

    # 중앙 원본 이미지의 네 꼭짓점
    original_corners = [
        (w, h),          # left-top
        (2 * w, h),      # right-top
        (2 * w, 2 * h),  # right-bottom
        (w, 2 * h),      # left-bottom
    ]

    rotated = tiled.rotate(
        angle,
        resample=Image.BICUBIC,
        expand=False,
        center=(cx, cy),
        fillcolor=fillcolor,
    )

    if not crop:
        return rotated

    # 3. 중앙 원본 영역의 회전 bbox 계산 후 crop
    left, top, right, bottom = _rotated_bbox(original_corners, cx, cy, angle)
    return rotated.crop((left, top, right, bottom))