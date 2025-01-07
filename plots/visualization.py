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
from tensorboard.compat.tensorflow_stub.io.gfile import exists
from torch.utils.data import DataLoader
from torch.utils.data.backward_compatibility import worker_init_fn

from dataset.OCT_dataset import OCTDataset
from dataset.datamodule_handler import get_data_modules
from models.mim import MIMWrapper
from transforms.apply_transforms import get_finetune_transformation, get_test_transformation
from transforms.masking import MIMTransform
from transforms.transformations import to_Tensor, to_PIL, colorJitter, gaussianBlur, rotation, sobelFilter, medianFilter
import matplotlib.pyplot as plt

from util.data_labels import get_merged_classes, get_full_classes
from util.utils import set_seed

if __name__ == "__main__":
    set_seed(42)
    load_dotenv(dotenv_path="../data/.env")
    comment = "mim_17"
    batch_size = 32
    img_size = 224
    epochs = 100
    devices = [1]
    mask_ratio = 0.5
    patch_size = 4
    mask_type = "center"
    device = "cuda:0"
    weights = [1]
    bins = [(mask_ratio, mask_ratio)]
    mim_transform = MIMTransform(img_size=img_size,
                                 model_patch_size=4,
                                 mask_patch_size=patch_size,
                                 mean=0.5,
                                 std=0.5,
                                 masking_type=mask_type,
                                 weights=weights,
                                 bins=bins,
                                 )

    masked_data_modules = get_data_modules(batch_size=batch_size,
                                           classes=get_full_classes(),
                                           filter_img=False,
                                           DS3_3mm=True,
                                           env_path="../data/.env",
                                           pretraining=False)
    # Merging train lists
    combined_train_list = (
            # masked_data_modules[0].data_train.img_paths
            # +
            # masked_data_modules[1].data_train.img_paths
            # +
            # masked_data_modules[2].data_train.img_paths
            # +
            masked_data_modules[3].data_train.img_paths
            # +
            # masked_data_modules[4].data_train.img_paths
            # +
            # masked_data_modules[5].data_train.img_paths
            # +
        # masked_data_modules[6].data_train.img_paths
        # +

        # masked_data_modules[7].data_train.img_paths +
            # masked_data_modules[7].data_unlabeled.img_paths +
            # masked_data_modules[8].data_unlabeled.img_paths +
            # masked_data_modules[0].data_val.img_paths +
            # masked_data_modules[1].data_val.img_paths +
            # masked_data_modules[2].data_val.img_paths +
            # masked_data_modules[3].data_val.img_paths +
            # masked_data_modules[4].data_val.img_paths +
            # masked_data_modules[5].data_val.img_paths +
            # masked_data_modules[7].data_val.img_paths
    )
    combined_train_dataset = OCTDataset(transform=mim_transform,
                                        data_dir="",
                                        img_paths=combined_train_list)
    train_masked_dataloader = DataLoader(combined_train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         num_workers=torch.cuda.device_count() * 2,
                                         pin_memory=True
                                         )

    load_path = os.path.join("../centralized/checkpoints", f"{comment}")
    load_path = os.path.join("../centralized/checkpoints", "archive","centralized","mim_ex56_100_0.5")
    print(load_path)
    model_path = glob.glob(os.path.join(load_path, f"mim*.ckpt"))
    mim = MIMWrapper.load_from_checkpoint(model_path[0],
                                          lr=2e-4,
                                          wd=0.05,
                                          min_lr=1e-5,
                                          epochs=epochs,
                                          warmup_lr=1e-6,
                                          warmup_epochs=10,
                                          weights=True,
                                          encoder="tiny"
                                          ).to(device)
    mim.model.encoder.to(device)
    mim.model.encoder.eval()
    
    finetune_transform = get_finetune_transformation(img_size)
    with torch.no_grad():
        for data in combined_train_dataset:
            mask = data[0][1]
            img = data[0][0]
            # Upscale the mask to the size of the image
            upscaled_mask = np.repeat(np.repeat(mask, 4, axis=0), 4, axis=1)

            # Ensure the upscaled mask has the same number of channels as the image
            # This step is a bit different for numpy arrays; we need to make sure the mask is correctly broadcasted
            # By default, numpy broadcasting should handle this, but if we need to explicitly match dimensions:
            upscaled_mask = upscaled_mask[None, :, :]  # Add an axis for channels if needed

            # Check if mask needs to be expanded to match image channels
            if img.shape[0] != upscaled_mask.shape[0]:
                upscaled_mask = np.repeat(upscaled_mask, img.shape[0], axis=0)


            # Convert mask to binary
            binary_mask = upscaled_mask > 0  # Adjust threshold as necessary
            # Prepare tensors for GPU processing
            tensor_mask = torch.tensor(binary_mask).to(device, dtype=torch.float32)
            img = img.to(device)
            mask = torch.tensor(mask).to(device).unsqueeze(dim=0)
            # mask = 1 - mask
            _, out = mim(img.unsqueeze(dim=0), mask)

            # Invert mask for selecting from the original image
            inv_mask = 1 - tensor_mask

            masked_image = img * inv_mask

            # result_image = masked_image.to(device) + out.squeeze().to(device) * inv_mask.to(device)
            result_image = out.squeeze().to(device)

            out = to_PIL()(result_image.cpu() * 0.5 + 0.5)
            img = to_PIL()(data[0][0]* 0.5 + 0.5)
            masked_image = to_PIL()(masked_image* 0.5 + 0.5)
            imgs = [img, rotation(img_size)(img), colorJitter(img_size)(img), medianFilter(img_size)(img),
                    gaussianBlur(img_size)(img), to_PIL()(inv_mask),
                    masked_image, out]
            imgs = [img,masked_image, out]
            plt.figure(figsize=(20, 10))  # Adjust the figure size as needed

            # Plot images in a 2x4 grid
            # for i in range(1, 9):
            #     plt.subplot(2, 4, i)
            #     plt.imshow(imgs[i - 1])  # Use eval to dynamically use img1, img2, ..., img8
            #     plt.axis('off')
            #     plt.title(f'Image {i}')
            #
            plt.rcParams['font.family'] = 'Thoma'
            plt.rcParams['font.size'] = 24

            plt.subplot(1, 3, 1)
            plt.imshow(imgs[0])  # Use eval to dynamically use img1, img2, ..., img8
            plt.axis('off')
            plt.title(f'(a) Input image')

            plt.subplot(1, 3, 2)
            plt.imshow(imgs[1])  # Use eval to dynamically use img1, img2, ..., img8
            plt.axis('off')
            plt.title(f'(b) Masked image')

            plt.subplot(1, 3, 3)
            plt.imshow(imgs[2])  # Use eval to dynamically use img1, img2, ..., img8
            plt.axis('off')
            plt.title(f'(c) Reconstruted image')

            plt.tight_layout()  # Adjust layout to not overlap images
            plt.savefig("sample4.png")
            exit(-1)
