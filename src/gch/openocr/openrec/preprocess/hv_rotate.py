import random

import numpy as np
from PIL import Image


class HV_90Rotate:
    """Rotate landscape/portrait samples by 90 degrees with independent probs.

    - If w/h > horizontal_ratio_threshold: regarded as horizontal, rotate
      clockwise by 90 with horizontal_rotate_prob.
    - If w/h < vertical_ratio_threshold: regarded as vertical, rotate
      counter-clockwise by 90 with vertical_rotate_prob.
    - Otherwise: keep unchanged.
    """

    def __init__(
        self,
        horizontal_ratio_threshold=1.0,
        vertical_ratio_threshold=1.0,
        horizontal_rotate_prob=0.5,
        vertical_rotate_prob=0.5,
        **kwargs,
    ):
        self.horizontal_ratio_threshold = float(horizontal_ratio_threshold)
        self.vertical_ratio_threshold = float(vertical_ratio_threshold)
        self.horizontal_rotate_prob = float(horizontal_rotate_prob)
        self.vertical_rotate_prob = float(vertical_rotate_prob)

        if self.horizontal_ratio_threshold <= 0.0:
            raise ValueError("horizontal_ratio_threshold must be > 0")
        if self.vertical_ratio_threshold <= 0.0:
            raise ValueError("vertical_ratio_threshold must be > 0")
        if not (0.0 <= self.horizontal_rotate_prob <= 1.0):
            raise ValueError("horizontal_rotate_prob must be between 0 and 1")
        if not (0.0 <= self.vertical_rotate_prob <= 1.0):
            raise ValueError("vertical_rotate_prob must be between 0 and 1")

    def _rotate_pil_cw90(self, img):
        return img.transpose(Image.Transpose.ROTATE_270)

    def _rotate_pil_ccw90(self, img):
        return img.transpose(Image.Transpose.ROTATE_90)

    @staticmethod
    def _rotate_np_cw90(img):
        return np.rot90(img, k=3)

    @staticmethod
    def _rotate_np_ccw90(img):
        return np.rot90(img, k=1)

    def __call__(self, data):
        img = data.get("image")
        if img is None:
            return data

        if isinstance(img, Image.Image):
            w, h = img.size
            if h <= 0 or w <= 0:
                return data
            ratio = float(w) / float(h)

            if ratio > self.horizontal_ratio_threshold:
                if random.random() < self.horizontal_rotate_prob:
                    data["image"] = self._rotate_pil_cw90(img)
            elif ratio < self.vertical_ratio_threshold:
                if random.random() < self.vertical_rotate_prob:
                    data["image"] = self._rotate_pil_ccw90(img)
            return data

        if isinstance(img, np.ndarray):
            if img.ndim < 2:
                return data
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                return data
            ratio = float(w) / float(h)

            if ratio > self.horizontal_ratio_threshold:
                if random.random() < self.horizontal_rotate_prob:
                    data["image"] = self._rotate_np_cw90(img)
            elif ratio < self.vertical_ratio_threshold:
                if random.random() < self.vertical_rotate_prob:
                    data["image"] = self._rotate_np_ccw90(img)
            return data

        return data
