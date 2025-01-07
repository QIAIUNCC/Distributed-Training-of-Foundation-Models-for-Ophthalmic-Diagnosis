import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

import transforms.transformations
from dataset.OCT_dataset import OCTDataset
from dataset.datamodule_handler import get_data_modules
from util.data_labels import get_full_classes


# Define function to calculate the average histogram of a group of images
def avg_hist(dataloader):
    # Initialize the total histogram
    total_hist = np.zeros((256,))

    for batch in dataloader:
        batch_hist = np.zeros((256,))
        for img_data in batch:
            img = cv2.imread(img_data, cv2.IMREAD_GRAYSCALE)

            # Calculate normalized histogram
            hist_img = cv2.calcHist([img], [0], None, [256], [0, 256])
            cv2.normalize(hist_img, hist_img)

            # Add to the batch histogram
            batch_hist += np.squeeze(hist_img)

        # Add batch histogram to the total histogram
        total_hist += batch_hist

        # Average the total histogram
    avg_hist = total_hist / len(dataloader)

    return avg_hist


load_dotenv(dotenv_path="../data/.env")
server_port = os.getenv('DATASET_PATH')
DATASET_PATH = os.getenv('DATASET_PATH')
kermany_classes, srinivasan_classes, oct500_classes, nur_classes, waterloo_classes, octdl_classes, uic_dr_classes = (
    get_full_classes())
NUM_WORKERS = 4 * 2

data_modules = get_data_modules(batch_size=128,
                                classes=get_full_classes(),
                                filter_img=False,
                                threemm=True)

# Configure plot font and size
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14  # Adjust as needed
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
# Plot histograms
fig, ax = plt.subplots(figsize=(10, 7))

for i, data_module in enumerate(data_modules[:-1]):
    # Calculate the average histograms
    combined_train = (data_module[0].data_train.img_paths +
                      data_module[0].data_val.img_paths +
                      data_module[0].data_test.img_paths
                      )
    combined_train = [data[0] for data in combined_train]
    dataloader = DataLoader(combined_train,
                            batch_size=128,
                            shuffle=False,
                            drop_last=False,
                            num_workers=torch.cuda.device_count() * 2,
                            pin_memory=True
                            )
    hist_group = avg_hist(dataloader)

    ax.plot(hist_group, color=colors[i % len(colors)], label=data_module[1])

ax.set_title('Average Normalized Histograms of Datasets', fontsize=18)  # Adjust as needed
ax.set_xlabel('Pixel Value', fontsize=14)  # Adjust as needed
ax.set_ylabel('Normalized Count', fontsize=14)  # Adjust as needed
ax.legend()

plt.tight_layout()
plt.show()
