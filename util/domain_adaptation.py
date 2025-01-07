import copy
import glob

import numpy as np
from PIL import Image
import torch
from dotenv import load_dotenv
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.data.backward_compatibility import worker_init_fn

from dataset.OCT_dataset import OCTDataset
from dataset.datamodule_handler import get_data_modules
from models.mim import SimMimWrapper
from transforms.apply_transforms import get_finetune_transformation, get_test_transformation, get_da_transformation
from transforms.masking import MIMTransform
from transforms.transformations import to_Tensor, to_PIL, colorJitter, gaussianBlur, rotation, sobelFilter, FastSVDNA
import matplotlib.pyplot as plt

from util.data_labels import get_merged_classes, get_full_classes
from util.utils import set_seed
from torchvision.transforms import transforms as T

if __name__ == "__main__":
    set_seed(9853)
    load_dotenv(dotenv_path="../data/.env")
    cls_batch_size = 1
    img_size = 224
    epochs = 100
    devices = [1]
    mim_ratio = 0.5

    kermany_classes, srinivasan_classes, oct500_classes, nur_classes, waterloo_classes, octd_classes, uic_classes = get_merged_classes()

    cls_data_modules = get_data_modules(batch_size=cls_batch_size,
                                        classes=get_merged_classes(),
                                        train_transform=get_da_transformation(img_size),
                                        test_transform=get_test_transformation(img_size))
    for datamodule, client_name, _ in cls_data_modules:
        for data in datamodule.train_dataloader():
            img = data[0][0]
            img = (img + 1) / 2 * 255
            out = to_PIL()(img.to(torch.uint8))
            plt.imshow(out, "gray")  # Use eval to dynamically use img1, img2, ..., img8
            plt.axis('off')
            plt.tight_layout()  # Adjust layout to not overlap images
            plt.savefig(client_name+"_da_sample.png", bbox_inches='tight', pad_inches=0)
            # plt.savefig(client_name+"_sample.png", bbox_inches='tight', pad_inches=0)
            break
