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
from dataset.data_module_handler import get_data_modules
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
    architecture = "mim_ex90"
    batch_size = 32
    cls_batch_size = 32
    img_size = 224
    epochs = 100
    devices = [1]
    mask_ratios = [0.95]

    (kermany_classes_full, srinivasan_classes_full, oct500_classes_full, nur_classes_full, waterloo_classes_full,
     octdl_classes_full, _, _) = get_full_classes()
    kermany_classes, srinivasan_classes, oct500_classes, nur_classes, waterloo_classes, octd_classes, _,_ = get_merged_classes()
    weights = [1]
    bins = [(0.75, 0.95)]
    mim_transform = MIMTransform(img_size, model_patch_size=4,
                                    mask_patch_size=16,
                                    mean=0.5,
                                    std=0.5,
                                    masking_type="random",
                                    top=0.15,
                                    bottom=0.15,
                                    bins=bins,
                                    weights=weights
                                    )
    cls_data_modules = get_data_modules(batch_size=cls_batch_size,
                                        classes=get_merged_classes(),
                                        train_transform=get_finetune_transformation(img_size),
                                        test_transform=get_test_transformation(img_size))
    masked_data_modules = get_data_modules(batch_size=batch_size,
                                           classes=get_full_classes(),
                                           train_transform=get_finetune_transformation(img_size),
                                           test_transform=get_test_transformation(img_size))
    # Merging train lists
    combined_train_list = (
            # masked_data_modules[0][0].data_train.img_paths +
            #                masked_data_modules[1][0].data_train.img_paths +
                           masked_data_modules[2].data_train.img_paths +
                           masked_data_modules[3].data_train.img_paths)
    combined_train_dataset = OCTDataset(transform=mim_transform, data_dir="",
                                        img_paths=combined_train_list)
    train_masked_dataloader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                         num_workers=torch.cuda.device_count() * 2, worker_init_fn=worker_init_fn)

    # Merging validation lists
    combined_val_list = (masked_data_modules[0].data_val.img_paths +
                         masked_data_modules[1].data_val.img_paths +
                         masked_data_modules[2].data_val.img_paths +
                         masked_data_modules[3].data_val.img_paths)
    combined_val_dataset = OCTDataset(transform=mim_transform, data_dir="",
                                      img_paths=combined_val_list)
    val_masked_dataloader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                       num_workers=torch.cuda.device_count() * 2, worker_init_fn=worker_init_fn)

    config = {"batch_size": batch_size, "epochs": epochs, "current_round": 1}
    # Simmim training
    load_path = os.path.join("../centralized/checkpoints", f"{architecture}")
    print(load_path)
    model_path = glob.glob(os.path.join(load_path, f"mim*.ckpt"))
    mim = MIMWrapper.load_from_checkpoint(model_path[0],
                                               device=devices[0],
                                               lr=5e-4,
                                               wd=0.05,
                                               min_lr=5e-7,
                                               patience=20,
                                               factor=0.5,
                                               epochs=epochs,
                                               warmup_lr=5e-7,
                                               warmup_epochs=epochs // 10,
                                                weight=True,
                                                encoder="base"
                                               ).to("cuda:1")
    mim.eval()
    mim.model.encoder.to("cuda:1")
    mim.model.encoder.eval()
    ### classification models
    param = {
        "wd": 1e-6,
        "lr": 3e-5,
        "beta1": 0.9,
        "beta2": 0.999,
    }
    config = {"batch_size": cls_batch_size, "epochs": epochs, "current_round": 1}
    finetune_transform = get_finetune_transformation(img_size)
    with torch.no_grad():
        for data in combined_train_dataset:
            img, mask = data[0]
            # Prepare tensors for GPU processing
            img = img.to("cuda:1")
            mask = torch.from_numpy(mask).to("cuda:1")
            _, out = mim.forward(img.unsqueeze(dim=0), mask.unsqueeze(dim=0))
            mask = mask.repeat_interleave(4, 0).repeat_interleave(4, 1)

            # result_image = masked_image.to("cuda:1") + out.squeeze().to("cuda:1") * inv_mask.to("cuda:1")
            result_image = out.squeeze().to("cuda:1")
            out = to_PIL()((result_image.cpu() + 1) /2 )
            masked_image = to_PIL()((torch.where(mask == 1.0,  img.new_tensor(1.0) , img) +1)/2)
            img = to_PIL()((img +1 )/2)
            imgs = [img, rotation(img_size)(img), colorJitter(img_size)(img), medianFilter(img_size)(img),
                    gaussianBlur(img_size)(img), to_PIL()((mask+1)/2),
                    masked_image, out]
            plt.figure(figsize=(20, 10))  # Adjust the figure size as needed

            # Plot images in a 2x4 grid
            for i in range(1, 9):
                plt.subplot(2, 4, i)
                plt.imshow(imgs[i - 1])  # Use eval to dynamically use img1, img2, ..., img8
                plt.axis('off')
                plt.title(f'Image {i}')

            plt.tight_layout()  # Adjust layout to not overlap images
            plt.show()
