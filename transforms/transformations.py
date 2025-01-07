import PIL.Image
import numpy as np
import torch
import albumentations as A
import torchvision.transforms.functional as TF
from PIL import ImageOps, ImageFilter, ImageDraw
from numpy import random
from scipy.ndimage import median_filter
from torchvision.transforms import transforms as T, GaussianBlur

from torchvision.transforms import InterpolationMode
import random
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image
from sklearn.utils.extmath import randomized_svd


def compute_cdf(hist):
    # Calculate the cumulative distribution function from the histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf.max())  # Normalize
    return cdf_normalized


def match_histograms(image, ref_hist):
    # Calculate the histogram for the image
    hist_img = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    cv2.normalize(hist_img, hist_img)

    # Compute the CDF for image and reference
    cdf_img = compute_cdf(hist_img)
    cdf_ref = compute_cdf(ref_hist)

    # Create a lookup table
    lookup_table = np.zeros(256)
    for i in range(256):
        diff_cdf = np.abs(cdf_ref - cdf_img[i])
        closest_index = np.argmin(diff_cdf)
        lookup_table[i] = closest_index

    # Apply the mapping to the image
    matched_image = cv2.LUT(image, lookup_table)
    return matched_image


class RandomCrop(object):
    """
    Take a random crop from the image.
    First the image or crop size may need to be adjusted if the incoming image
    is too small...
    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image
    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """

    def __init__(self, crop_size=(400, 400), resize=(128, 128), nopad=True):

        # if isinstance(crop_size, numbers.Number):
        #     self.size = (int(crop_size), int(crop_size))
        # else:
        #     self.size = crop_size
        self.crop_size = crop_size
        self.resize = resize
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, centroid=None):
        w, h = img.size
        # ASSUME H, W
        th, tw = self.crop_size
        if w == tw and h == th:
            return img

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                w, h = img.size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)).resize(size=self.resize, resample=PIL.Image.LANCZOS)


class SobelFilter():
    def __call__(self, img):
        img_x = img.filter(ImageFilter.FIND_EDGES)
        img_y = img.transpose(
            PIL.Image.FLIP_LEFT_RIGHT).filter(ImageFilter.FIND_EDGES).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        return PIL.Image.blend(img_x, img_y, alpha=0.5)


class CustomRotation:
    def __init__(self, angles=None):
        if angles is None:
            angles = [0, 90, 180, 270]
        self.angles = angles

    def __call__(self, image):
        angle = random.choice(self.angles)
        return TF.rotate(image, angle, expand=True)


class MatchHistogramsTransform:
    def __init__(self, ref_hist=None):
        if ref_hist is None:
            ref_hist = np.load("../utils/avg_hist.npy")
        self.ref_hist = ref_hist

    def __call__(self, image):
        # Assuming the input image is a PIL image
        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Apply the histogram matching
        matched_image = match_histograms(image_np, self.ref_hist)

        # Convert numpy array back to PIL image
        matched_image_pil = PIL.Image.fromarray(matched_image).convert("L")
        return matched_image_pil


class UnsharpMaskTransform(object):
    def __init__(self, radius=2, percent=150, threshold=3):
        """
        Initialize the Unsharp Mask filter parameters.
        :param radius: The radius of the blur filter.
        :param percent: The percentage of the edge enhancement.
        :param threshold: The threshold for the filter.
        """
        self.radius = radius
        self.percent = percent
        self.threshold = threshold

    def __call__(self, img):
        """
        Apply the Unsharp Mask filter to the input image.
        :param img: PIL image to be sharpened.
        :return: Sharpened PIL image.
        """
        return img.filter(ImageFilter.UnsharpMask(
            radius=self.radius, percent=self.percent, threshold=self.threshold))


class ZScoreNormalization:
    """Apply Z-score normalization to an image."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image to be normalized.

        Returns:
            Tensor: Z-score normalized image.
        """
        if self.mean is None or self.std is None:
            # Calculate mean and std if not provided
            self.mean = img.mean()
            self.std = img.std()
        return (img - self.mean) / (self.std + 1e-8)  # Adding a small value to avoid division by zero

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def img_read(img):
    return np.array(img)


class SVDNA:
    def __init__(self, target_path, img_size, k=50, histo_matching_degree=1):
        self.k = k
        target_img = np.asarray(T.Resize((img_size, img_size),
                                         interpolation=InterpolationMode.BICUBIC)
                                (PIL.Image.open(target_path).convert("L")))
        u_target, s_target, vh_target = np.linalg.svd(target_img, full_matrices=False)
        thresholded_singular_target = s_target
        thresholded_singular_target[0:k] = 0
        self.target_style = np.array([np.dot(u_target, np.dot(np.diag(thresholded_singular_target), vh_target))])
        self.transformHist = A.Compose([
            A.HistogramMatching([target_img], blend_ratio=(histo_matching_degree, histo_matching_degree),
                                read_fn=img_read,
                                p=1)
        ])

    def __call__(self, img):
        u_source, s_source, vh_source = np.linalg.svd(img, full_matrices=False)
        thresholded_singular_source = s_source
        thresholded_singular_source[self.k:] = 0
        content_src = np.array([np.dot(u_source, np.dot(np.diag(thresholded_singular_source), vh_source))])
        noise_adapted_im = content_src + self.target_style
        noise_adapted_im_clipped = np.squeeze(noise_adapted_im).clip(0, 255).astype(np.uint8)
        transformed = self.transformHist(image=noise_adapted_im_clipped)
        return PIL.Image.fromarray(transformed["image"])


class FastSVDNA:
    def __init__(self, target_path, img_size, k=50, histo_matching_degree=1):
        self.k = k
        target_img = (T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC)
                      (PIL.Image.open(target_path).convert("L")))
        target_img = T.ToTensor()(target_img)
        # target_img = T.Normalize(mean=0.5, std=0.5)(target_img)
        u, s, v = torch.svd(target_img)
        thresholded_singular_target = s.squeeze()
        thresholded_singular_target[0:k] = 0
        self.target_style = torch.matmul(torch.matmul(u.squeeze(), torch.diag(thresholded_singular_target)),
                                         v.squeeze().T)
        self.transformHist = A.Compose([
            A.HistogramMatching([target_img.squeeze()], blend_ratio=(histo_matching_degree, histo_matching_degree),
                                read_fn=img_read,
                                p=1)
        ])

    def __call__(self, img):
        # u_source, s_source, vh_source = randomized_svd(np.asarray(img), n_components=50, random_state=42)
        u, s, v = torch.svd(img)
        thresholded_singular = s.squeeze()
        thresholded_singular[self.k:] = 0
        content_src = torch.matmul(torch.matmul(u.squeeze(), torch.diag(thresholded_singular)), v.squeeze().T)
        noise_adapted_im = content_src + self.target_style
        noise_adapted_im_clipped = np.asarray(noise_adapted_im.clip(0, 1)).astype(np.float32)
        transformed = self.transformHist(image=noise_adapted_im_clipped)
        return T.ToPILImage()((transformed["image"] * 255).astype(np.uint8))


class IdealCircularLPFTransform(object):
    def __init__(self, cutoff_frequency):
        self.cutoff_frequency = cutoff_frequency

    def forward(self, img):
        img_tensor = to_tensor(img).float()
        # Step 1: Perform DFT
        img_fft = torch.fft.fftshift(torch.fft.fft2(img_tensor))

        # Step 2: Apply the ideal circular LPF
        _, H, W = img_fft.shape
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        center_y, center_x = H // 2, W // 2
        dist_from_center = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        mask = dist_from_center <= self.cutoff_frequency
        img_fft_filtered = img_fft * mask.float()

        # Step 3: Perform inverse DFT and take the real part
        img_filtered = torch.fft.ifft2(torch.fft.ifftshift(img_fft_filtered)).real

        # Convert tensor back to PIL image for torchvision compatibility
        img_filtered = to_pil_image(img_filtered.clamp(0, 1))
        return img_filtered

    def __call__(self, img):
        return self.forward(img)


def apply_median_filter(image, filter_size):
    """
    Apply a median filter to a grayscale image.

    Parameters:
    image (numpy array): Input grayscale image.
    filter_size (int): Size of the median filter (must be an odd number).

    Returns:
    numpy array: Image after applying the median filter.
    """
    if filter_size % 2 == 0:
        raise ValueError("Filter size must be an odd number.")

    # Apply the median filter
    filtered_image = median_filter(image, size=filter_size)

    return filtered_image


class MedianFilterTransform:
    def __init__(self, filter_size):
        if filter_size % 2 == 0:
            raise ValueError("Filter size must be an odd number.")
        self.filter_size = filter_size

    def __call__(self, image):
        # Convert the image to numpy array
        image_np = np.array(image)
        print(image_np.shape)

        # Convert the image to a numpy array
        image_np = np.array(image)

        # Apply the median filter using scipy
        filtered_image_np = median_filter(image_np, size=self.filter_size)

        # Convert back to PIL image
        filtered_image = PIL.Image.fromarray(filtered_image_np)
        return filtered_image


def rotation(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
        CustomRotation(angles=[90, 180, 270]),
        T.Grayscale(3),
    ])


def colorJitter(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
        T.ElasticTransform(alpha=(50.0, 250.0), sigma=(5.0, 10.0)),
        T.Grayscale(3),
    ])


def sobelFilter(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
        SobelFilter(),
        T.Grayscale(3),
    ])


def gaussianBlur(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
        GaussianBlur(kernel_size=int(5), sigma=(0.25, 1.75)),
        T.Grayscale(3),
    ])


def medianFilter(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
        MedianFilterTransform(filter_size=3),
        T.Grayscale(3)
    ])


def to_PIL():
    return T.Compose([T.ToPILImage(),
                      T.Grayscale(3)
                      ])


def to_Tensor():
    return T.ToTensor()

def ensure_three_channels(img):
    """Convert an image to 3 channels (RGB)."""
    if img.mode != 'RGB':
        img = img.convert("RGB")
    return img