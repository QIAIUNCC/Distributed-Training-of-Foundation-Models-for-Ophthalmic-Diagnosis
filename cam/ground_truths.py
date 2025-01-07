import os.path

import numpy as np
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode
from dataset.datamodule_handler import get_data_modules
import matplotlib.pyplot as plt
from transforms.apply_transforms import get_finetune_transformation, get_test_transformation
from util.data_labels import get_merged_classes
from util.get_models import get_mim_model, get_cls_model
from util.utils import set_seed
import torch
if __name__ == '__main__':
    set_seed(42)
    mim_architecture = "centralized/mim_ex56"
    img_size = 128
    batch_size = 1
    mim = get_mim_model(mim_architecture)
    mim.eval()
    param = {
        "wd": 1e-6,
        "lr": 3e-5,
        "beta1": 0.9,
        "beta2": 0.999,
    }
    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_merged_classes(),
                                    train_transform=get_finetune_transformation(img_size),
                                    test_transform=get_test_transformation(img_size,
                                                                           apply_adaptation=False),
                                    threemm=True,
                                    env_path="../data/.env")
    resize = T.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC)

    imgs = {"DS1": 3750, "DS2": 750, "DS3": 4428, "DS4": 1826, "DS5": 6, "DS6":28, "DS7": 28}
    for data_module, client_name, classes in data_modules:
        j = 1
        for data in data_module.test_dataloader():
            if j == imgs[client_name]:
                images = data[0]
                label = data[1]
                img = (images + 1) / 2 * 255
                img = img.to(torch.uint8)
                plt.imshow(resize(img).squeeze(0)[1, :, :],
                           "gray")  # Make sure the image is in RGB format if it's a color image
                plt.axis('off')  # Hide the axis

                plt.savefig(os.path.join(f"./gt",
                                         client_name, str(j) + ".png"),
                            bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f"image {j} has been proceed!")
                break
            j += 1
