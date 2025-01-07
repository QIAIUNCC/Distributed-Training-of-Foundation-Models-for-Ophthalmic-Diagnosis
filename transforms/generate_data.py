import torch
from dotenv import load_dotenv
import os
from torch.utils.data import DataLoader
from torch.utils.data.backward_compatibility import worker_init_fn
from tqdm import tqdm
from dataset.OCT_dataset import OCTDataset
from dataset.data_module_handler import get_data_modules
from PIL import Image
from torchvision.transforms import transforms as T, InterpolationMode

from transforms.transformations import FastSVDNA
from util.data_labels import get_full_classes

if __name__ == "__main__":
    load_dotenv(dotenv_path="../data/.env")
    batch_size = 128
    img_size = 256
    (kermany_classes_full, srinivasan_classes_full, oct500_classes_full, nur_classes_full, waterloo_classes_full,
     octdl_class_full, uic_class_full) = get_full_classes()

    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_full_classes(),
                                    filter_img=False)
    # Merging train lists
    combined_data = (
            data_modules[0][0].data_train.img_paths +
            data_modules[0][0].data_val.img_paths +
            data_modules[0][0].data_test.img_paths +
            data_modules[1][0].data_train.img_paths +
            data_modules[1][0].data_val.img_paths +
            data_modules[1][0].data_test.img_paths +
            data_modules[-1][0].data_train.img_paths +
            data_modules[-1][0].data_val.img_paths +
            data_modules[-1][0].data_test.img_paths +
            data_modules[2][0].data_train.img_paths +
            data_modules[3][0].data_train.img_paths +
            data_modules[4][0].data_train.img_paths +
            data_modules[5][0].data_train.img_paths +
            data_modules[2][0].data_val.img_paths +
            data_modules[3][0].data_val.img_paths +
            data_modules[4][0].data_val.img_paths +
            data_modules[5][0].data_val.img_paths +
            data_modules[2][0].data_test.img_paths +
            data_modules[3][0].data_test.img_paths +
            data_modules[4][0].data_test.img_paths +
            data_modules[5][0].data_test.img_paths

    )
    print(len(data_modules[2][0].data_train.img_paths) + len(data_modules[2][0].data_val.img_paths) + len(
        data_modules[2][0].data_test.img_paths))
    combined_dataset = OCTDataset(transform=None,
                                  data_dir="",
                                  img_paths=combined_data)
    dataloader = DataLoader(combined_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=torch.cuda.device_count() * 2,
                            worker_init_fn=worker_init_fn)
    trans = T.Compose([
        T.Resize((img_size, img_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        # T.Normalize(mean=0.5, std=0.5),
        FastSVDNA(target_path="NORMAL-36734-8.jpeg", img_size=img_size),
        # UnsharpMaskTransform(radius=2, percent=150, threshold=3),
    ])
    for i in tqdm(combined_data):
        path = i[0]
        new_path = path.replace("/OCT/", "/newdata/", 1)
        if os.path.exists(new_path):
            continue
        elif not os.path.exists("/".join(new_path.split("/")[:-1])):
            print("/".join(new_path.split("/")[:-1]))
            os.makedirs("/".join(new_path.split("/")[:-1]), exist_ok=True)
        img = Image.open(path).convert("L")
        img = trans(img)
        # Check if the directory exists, and if not, create it
        img.save(new_path)
