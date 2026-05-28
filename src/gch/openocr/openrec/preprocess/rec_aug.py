import random
from typing import Any

import cv2
import numpy as np
from PIL import Image


class Flip:
    """Probabilistic image flip augmentation.

    Args:
        p (float): Probability to apply flip. Must be in [0, 1].
        direction (str): Flip direction.
            - "horizontal" (default): left-right flip
            - "vertical": up-down flip
            - "both": randomly choose horizontal or vertical
    """

    def __init__(self, p: float = 0.5, direction: str = "horizontal", **kwargs):
        self.p = float(p)
        if self.p < 0.0 or self.p > 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")

        direction = str(direction).lower()
        valid = {"horizontal", "vertical", "both"}
        if direction not in valid:
            raise ValueError(
                f"direction must be one of {sorted(valid)}, got {direction}")
        self.direction = direction

    def _get_direction(self) -> str:
        if self.direction == "both":
            return random.choice(["horizontal", "vertical"])
        return self.direction

    def _flip_numpy(self, image: np.ndarray, direction: str) -> np.ndarray:
        if direction == "horizontal":
            return cv2.flip(image, 1)
        return cv2.flip(image, 0)

    def _flip_pil(self, image: Image.Image, direction: str) -> Image.Image:
        if direction == "horizontal":
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image.transpose(Image.FLIP_TOP_BOTTOM)

    def __call__(self, data: dict) -> dict:
        image: Any = data["image"]
        if random.random() >= self.p:
            return data

        direction = self._get_direction()

        if isinstance(image, np.ndarray):
            data["image"] = self._flip_numpy(image, direction)
            return data

        if isinstance(image, Image.Image):
            data["image"] = self._flip_pil(image, direction)
            return data

        raise TypeError(
            f"Flip expects image as np.ndarray or PIL.Image, got {type(image)}")
