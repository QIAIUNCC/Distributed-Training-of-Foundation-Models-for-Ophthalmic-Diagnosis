import cv2
import numpy as np
from torchvision.transforms import transforms as T, GaussianBlur, InterpolationMode

from transforms.apply_transforms import get_pretrain_transformation
from transforms.transformations import ZScoreNormalization, UnsharpMaskTransform, SobelFilter, FastSVDNA
import PIL.Image
import torchvision.transforms.functional as F
from numpy import random

from util.utils import random_choice_with_weighted_distribution


class DynamicMaskGenerator:
    def __init__(self, img_size, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.img_size = img_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.img_size % self.mask_patch_size == 0, "Input size must be divisible by mask patch size"
        assert self.mask_patch_size % self.model_patch_size == 0, "Mask patch size must be divisible by model patch size"

        self.rand_size = self.img_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.floor(self.token_count * self.mask_ratio))

    def find_active_region(self, img):
        # Threshold the image to focus on significant areas
        _, thresh = cv2.threshold(np.array(img), 30, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to remove noise and fill gaps
        kernel = np.ones((5, 5), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find the first and last non-zero pixel rows
        active = np.any(clean > 0, axis=1)
        top_index = np.argmax(active)  # First true index from the top
        bottom_index = len(active) - np.argmax(active[::-1])  # First true index from the bottom

        # Convert pixel indices to patch indices
        top_patch = top_index // self.mask_patch_size
        bottom_patch = bottom_index // self.mask_patch_size

        return top_patch, bottom_patch

    def __call__(self, img):
        top_patch, bottom_patch = self.find_active_region(img)
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)

        # Only include indices that are not in the top or bottom exclusion zones
        valid_indices = [idx for idx in mask_idx if
                         idx // self.rand_size >= top_patch and idx // self.rand_size < bottom_patch]

        mask[valid_indices] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class CenterMaskGenerator:
    def __init__(self, img_size=192, mask_patch_size=32, model_patch_size=4, top=0.25, bottom=0.25,
                 bins=None, weights=None):
        if weights is None:
            weights = [0.1, 0.1, 0.2, 0.6]
        if bins is None:
            bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.75)]
        self.img_size = img_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.top = top
        self.bottom = bottom

        assert self.img_size % self.mask_patch_size == 0, "Input size must be divisible by mask patch size"
        assert self.mask_patch_size % self.model_patch_size == 0, "Mask patch size must be divisible by model patch size"

        self.rand_size = self.img_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2

        self.bins = bins
        self.cumulative_weights = [sum(weights[:i + 1]) for i in range(len(weights))]

    def __call__(self, img):
        self.mask_count = int(np.floor(self.token_count * random_choice_with_weighted_distribution(self.bins, self.cumulative_weights)))
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)

        # Number of tokens to exclude from the top
        exclude_top = int(self.token_count * self.top)
        # Index from which to start including tokens again
        exclude_bottom = self.token_count - int(self.token_count * self.bottom)

        # Exclude indices from the top 25% and bottom 25%
        valid_indices = [idx for idx in mask_idx if not (idx < exclude_top or idx >= exclude_bottom)]

        mask[valid_indices] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class RandomMaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4,
                 bins=None, weights=None):
        if weights is None:
            weights = [0.1, 0.1, 0.2, 0.6]
        if bins is None:
            bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.75)]
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size

        assert self.input_size % self.mask_patch_size == 0, "Input size must be divisible by mask patch size"
        assert self.mask_patch_size % self.model_patch_size == 0, "Mask patch size must be divisible by model patch size"

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.bins = bins
        self.cumulative_weights = [sum(weights[:i + 1]) for i in range(len(weights))]

    def __call__(self, img):
        self.mask_count = int(
            np.floor(self.token_count * random_choice_with_weighted_distribution(self.bins, self.cumulative_weights)))

        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class MIMTransform:
    def __init__(self, img_size, model_patch_size, mask_patch_size, transform,
                 masking_type="center", top=0.25, bottom=0.25, bins=None, weights=None):
        if weights is None:
            weights = [0.1, 0.1, 0.2, 0.6]
        if bins is None:
            bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.75)]
        self.transform_img = transform
        if masking_type == "center":
            self.mask_generator = CenterMaskGenerator(
                img_size=img_size,
                mask_patch_size=mask_patch_size,
                model_patch_size=model_patch_size,
                bins=bins,
                weights=weights,
                top=top,
                bottom=bottom
            )
        elif masking_type == "random":
            self.mask_generator = RandomMaskGenerator(
                input_size=img_size,
                mask_patch_size=mask_patch_size,
                model_patch_size=model_patch_size,
                bins=bins,
                weights=weights,
            )

        self.img_size = img_size

    def __call__(self, img):
        mask = self.mask_generator(img)
        mask = PIL.Image.fromarray(mask.astype(np.uint8))
        # Apply the same rotation to both image and mask
        rotation_angle = int(random.choice([0, 90, 180, 270]))
        img = F.rotate(img, rotation_angle, expand=True, interpolation=InterpolationMode.BICUBIC)
        mask = F.rotate(mask, rotation_angle, expand=True,
                        interpolation=InterpolationMode.NEAREST)  # Use NEAREST for masks to avoid interpolation artifacts
        mask = np.array(mask)
        # Now apply the rest of the transformations, including resize
        img = T.Resize((self.img_size, self.img_size), InterpolationMode.BICUBIC)(img)
        img = self.transform_img(img)
        return img, mask
