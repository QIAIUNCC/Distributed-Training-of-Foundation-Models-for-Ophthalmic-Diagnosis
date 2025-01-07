from torchvision.transforms import InterpolationMode

from transforms.transformations import ZScoreNormalization, UnsharpMaskTransform, FastSVDNA, SobelFilter, CustomRotation, ensure_three_channels
from torchvision.transforms import transforms as T, GaussianBlur
import torchvision

def  get_pretrain_transformation(img_size, mean=0.5, std=0.5, apply_adaptation=False,
                                 dad_img_path="../transforms/NORMAL-36734-8.jpeg"):
    if apply_adaptation:
        return T.Compose([
            T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
            T.ToTensor(),
            FastSVDNA(target_path=dad_img_path, img_size=img_size),
            T.RandomHorizontalFlip(p=0.2),
            T.RandomVerticalFlip(p=0.2),
            T.RandomApply([UnsharpMaskTransform(radius=2, percent=150, threshold=3), ], p=0.2),
            T.RandomApply([T.ColorJitter(0.5, 0.5)], p=0.15),
            T.RandomApply([GaussianBlur(kernel_size=int(5), sigma=(0.75, 1.5))], p=0.15),
            T.RandomApply([SobelFilter()], p=0.15),
            T.Grayscale(3),
            T.ToTensor(),
            T.Normalize((mean,), (std,))
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.2),
            T.RandomVerticalFlip(p=0.2),
            T.RandomApply([UnsharpMaskTransform(radius=2, percent=150, threshold=3), ], p=0.2),
            T.RandomApply([T.ColorJitter(0.5, 0.5)], p=0.15),
            T.RandomApply([GaussianBlur(kernel_size=int(5), sigma=(0.75, 1.5))], p=0.15),
            T.RandomApply([SobelFilter()], p=0.15),
            T.Grayscale(3),
            T.ToTensor(),
            T.Normalize((mean,), (std,))
        ])


def  get_RGB_pretrain_transformation(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
        torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.IMAGENET), 
        T.Lambda(ensure_three_channels),  
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
   


def get_finetune_transformation(img_size, mean=0.5, std=0.5, apply_adaptation=False, dad_img_path="../NORMAL-36734-8.jpeg"):
    if apply_adaptation:
        return T.Compose([
            T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
            CustomRotation(angles=[0, 90, 180, 270]),
            T.ToTensor(),
            FastSVDNA(target_path=dad_img_path, img_size=img_size),
            T.RandomHorizontalFlip(p=0.2),
            T.RandomVerticalFlip(p=0.2),
            T.RandomApply([UnsharpMaskTransform(radius=2, percent=150, threshold=3)], p=0.2),
            T.RandomApply([T.ColorJitter(0.5, 0.5)], p=0.2),
            T.RandomApply([GaussianBlur(kernel_size=int(5), sigma=(0.75, 1.5))], p=0.2),
            T.RandomApply([SobelFilter()], p=0.2),
            T.Grayscale(3),
            T.ToTensor(),
            T.Normalize((mean,), (std,))
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
            CustomRotation(angles=[0, 90, 180, 270]),
            T.RandomHorizontalFlip(p=0.2),
            T.RandomVerticalFlip(p=0.2),
            T.RandomApply([UnsharpMaskTransform(radius=2, percent=150, threshold=3)], p=0.2),
            T.RandomApply([T.ColorJitter(0.5, 0.5)], p=0.2),
            T.RandomApply([GaussianBlur(kernel_size=int(5), sigma=(0.75, 1.5))], p=0.2),
            T.RandomApply([SobelFilter()], p=0.2),
            T.Grayscale(3),
            T.ToTensor(),
            T.Normalize((mean,), (std,))
        ])




def get_test_transformation(img_size, mean=0.5, std=0.5, apply_adaptation=False, dad_img_path="../NORMAL-36734-8.jpeg"):
    if apply_adaptation:
        return T.Compose([
            T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
            T.ToTensor(),
            FastSVDNA(target_path=dad_img_path, img_size=img_size),
            UnsharpMaskTransform(radius=2, percent=150, threshold=3),
            T.Grayscale(3),
            T.ToTensor(),
            T.Normalize((mean,), (std,))
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
            UnsharpMaskTransform(radius=2, percent=150, threshold=3),
            T.Grayscale(3),
            T.ToTensor(),
            T.Normalize((mean,), (std,))
        ])


def get_da_transformation(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
        T.ToTensor(),
        FastSVDNA(target_path="./NORMAL-36734-8.jpeg", img_size=img_size, k=100),
        UnsharpMaskTransform(radius=2, percent=150, threshold=3),
        T.Grayscale(1),
        T.ToTensor(),
        T.Normalize(std=0.5, mean=0.5)
    ])
