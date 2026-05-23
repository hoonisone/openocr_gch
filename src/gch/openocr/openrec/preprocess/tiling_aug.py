import math
import random
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image


RotateRange = Union[
    float,
    int,
    Tuple[float, float],
    List[float],
    List[List[float]],
    List[Tuple[float, float]],
]


class TilingAug:
    """Apply 3x3 tiling + rotation + center-region crop augmentation."""

    def __init__(
        self,
        rotate_range: RotateRange = ((-25.0, -20.0), (20.0, 25.0)),
        rotate_prob: float = 1.0,
        crop: bool = True,
        fillcolor: Optional[Tuple[int, int, int]] = None,
        aug_prob: float = 0.5,
        **kwargs,
    ):
        self.rotate_range = rotate_range
        self.rotate_prob = float(rotate_prob)
        self.crop = bool(crop)
        self.fillcolor = fillcolor
        self.aug_prob = float(aug_prob)

    def _sample_angle(self) -> float:
        rr = self.rotate_range

        if isinstance(rr, (int, float)):
            value = abs(float(rr))
            return random.uniform(-value, value)

        if not isinstance(rr, (list, tuple)):
            raise ValueError(
                "rotate_range must be a number, [min, max], or [[min, max], ...]"
            )

        if len(rr) == 2 and all(isinstance(v, (int, float)) for v in rr):
            low = float(rr[0])
            high = float(rr[1])
            if low > high:
                raise ValueError(f"Invalid rotate_range: [{low}, {high}]")
            return random.uniform(low, high)

        if len(rr) == 0:
            raise ValueError("rotate_range cannot be empty")

        for interval in rr:
            if (
                not isinstance(interval, (list, tuple))
                or len(interval) != 2
                or not all(isinstance(v, (int, float)) for v in interval)
            ):
                raise ValueError(
                    "Each rotate_range interval must be [min, max], e.g. [[-20, -10], [10, 20]]"
                )
            low = float(interval[0])
            high = float(interval[1])
            if low > high:
                raise ValueError(f"Invalid rotate interval: [{low}, {high}]")

        selected = random.choice(rr)
        return random.uniform(float(selected[0]), float(selected[1]))

    def _augment_pil(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB")
        w, h = img.size

        tiled = Image.new("RGB", (w * 3, h * 3))
        for row in range(3):
            for col in range(3):
                tiled.paste(img, (col * w, row * h))

        cx, cy = 1.5 * w, 1.5 * h
        original_corners = [(w, h), (2 * w, h), (2 * w, 2 * h), (w, 2 * h)]

        if random.random() < self.rotate_prob:
            angle = self._sample_angle()
        else:
            angle = 0.0

        rotated = tiled.rotate(
            angle,
            resample=Image.BICUBIC,
            expand=False,
            center=(cx, cy),
            fillcolor=self.fillcolor,
        )

        if not self.crop:
            return rotated

        theta = math.radians(angle)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rotated_corners = []
        for x, y in original_corners:
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
        return rotated.crop((left, top, right, bottom))

    def __call__(self, data):
        if random.random() > self.aug_prob:
            return data

        img = data["image"]
        if isinstance(img, Image.Image):
            data["image"] = self._augment_pil(img)
            return data

        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
            data["image"] = np.array(self._augment_pil(pil_img))
            return data

        return data
