import copy
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from sklearn.metrics import roc_curve, auc
from dotenv import load_dotenv
from torch import nn

import lightning.pytorch as pl
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.colors as mcolors
from torchvision.models import Swin_V2_S_Weights, Swin_V2_T_Weights

from dataset.datamodule_handler import get_data_modules
from models.base import BaseNet
from models.cls import ClsMIM, MLP
from models.mim import MIMWrapper
from transforms.apply_transforms import get_finetune_transformation, get_test_transformation
from util.data_labels import get_full_classes, get_merged_classes
from util.utils import set_seed
from torchmetrics import AUROC
import torch.nn.functional as F
import torch





def plot_auc(colors, labels, preds, auroc_values, line_style, classes):
    classes = {"Control": 0,
               "Mild": 1,
               "Moderate": 2,
               "Severe": 3,
               }
    for i in range(len(classes)):
        # Compute ROC curve using sklearn's roc_curve
        class_name = list(classes.keys())[i]
        color = colors.get(class_name, 'tab:gray')
        fpr, tpr, _ = roc_curve(labels == i, preds[:, i])
        roc_auc = auroc_values[i].item()  # Get the AUROC value for the i-th class
        if not np.any(fpr) or not np.any(tpr) or roc_auc == 0:
            continue
        # Format roc_auc value to remove trailing zeros
        roc_auc_str = f'{roc_auc:.4f}'.rstrip('0').rstrip('.')
        plt.plot(fpr, tpr, color=color, lw=2, linestyle=line_style,
                 label=f'{list(classes.keys())[i]} (area = {roc_auc_str})')


def get_local_model(architecture, classes, client_name, root="../centralized"):
    encoder = nn.Sequential(torchvision.models.swin_v2_t(),
                            MLP(1000, len(classes)))
    print(os.path.join(root, "checkpoints", "archive", architecture, client_name, "*.ckpt"))
    print(architecture)
    local_path = glob.glob(
        os.path.join(root, "checkpoints", "archive", architecture, client_name, "*.ckpt"))

    model = BaseNet.load_from_checkpoint(local_path[0], classes=classes, lr=3e-5, wd=1e-6, encoder=encoder)
    return model


if __name__ == "__main__":
    set_seed(42)
    load_dotenv(dotenv_path="../data/.env")
    architecture_local = "local/ex14_l_swinv2_t_100"
    architecture_mim = "centralized/mim_ex56_100_0.5"
    architecture_central = "centralized/mim_ex56_cls_100_0.5"
    cls_batch_size = 128
    img_size = 128
    epochs = 100
    devices = [0]

    ds7 = get_data_modules(batch_size=cls_batch_size,
                           classes=get_merged_classes(),
                           train_transform=get_finetune_transformation(img_size),
                           test_transform=get_test_transformation(img_size,
                                                                  apply_adaptation=False),
                           threemm=False,
                           env_path="../data/.env")[-3]
    param = {
        "wd": 1e-6,
        "lr": 3e-5,
        "beta1": 0.9,
        "beta2": 0.999,
        "step_size": len(ds7.train_dataloader())
    }

    # local = get_local_model(architecture=architecture_local, classes=ds7.classes, client_name=ds7.dataset_name)
    mim = get_mim_model(architecture_mim, epochs=epochs)
    central = get_cls_model(architecture_central, copy.deepcopy(mim.model.encoder), ds7.classes, ds7.dataset_name,
                            param)
    da = ""
    fl_ex = "centralized"
    # architecture_fl = f"fl_ex{fl_ex}_100_0.5"
    mim = MIMWrapper(lr=1.5e-3,
                           wd=0.05,
                           min_lr=1e-5,
                           patience=10,
                           epochs=epochs,
                           warmup_lr=1e-5,
                           warmup_epochs=10,
                           gamma=0.1,
                           device="cuda:" + str(devices[0]),
                           weights=False
                           )
    # state_dict = torch.load(f'../fl/ex{fl_ex}.pth')
    # # Adjust the keys by removing the 'model.' prefix
    # adjusted_state_dict = {'model.' + key: value for key, value in state_dict.items()}
    # mim.load_state_dict(adjusted_state_dict)
    #
    # fl = get_cls_model(architecture_fl, copy.deepcopy(mim.model.encoder), ds7.classes, ds7.dataset_name, param)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=devices,
        log_every_n_steps=1,
        deterministic=True
    )
    n_classes = len(ds7.classes)
    test_loader = ds7.filtered_test_dataloader(ds7.classes)
    # test_results_local = trainer.test(local, dataloaders=ds7.test_dataloader(), verbose=False)
    test_results_central = trainer.test(central, dataloaders=ds7.test_dataloader(), verbose=False)
    # test_results_fl = trainer.test(fl, dataloaders=ds7.test_dataloader(), verbose=False)
    # print(test_results_fl)
    # Convert logits to probabilities
    # cls_local_probs = F.softmax(local.preds, dim=1).cpu().numpy()
    cls_central_probs = F.softmax(central.preds, dim=1).cpu().numpy()
    # cls_fl_probs = F.softmax(fl.preds, dim=1).cpu().numpy()

    # Generate adaptive colors based on the number of classes
    colors = {
        "Control": "tab:blue",
        "Mild": "tab:orange",
        "Moderate": "tab:green",
        "Severe": "tab:red",
    }
    plt.rcParams['font.family'] = 'Thoma'
    plt.rcParams['font.size'] = 17
    # Plot all ROC curves with predefined values from torchmetrics
    plt.figure(figsize=(8, 6))
    # Initialize the AUROC metric
    # auroc_metric_local = AUROC(task="multiclass", num_classes=n_classes, average=None)
    # # Compute AUROC for each class
    # auroc_values_local = auroc_metric_local(torch.tensor(cls_local_probs), local.labels.cpu())
    # plot_auc(colors, local.labels.cpu(), cls_local_probs, auroc_values_local, line_style='-', classes=ds7.classes)
    # auroc_metric_local.reset()

    auroc_metric_central = AUROC(task="multiclass", num_classes=n_classes, average=None)
    auroc_values_central = auroc_metric_central(torch.tensor(cls_central_probs), central.labels.cpu())
    plot_auc(colors, central.labels.cpu(), cls_central_probs, auroc_values_central, line_style='-',
             classes=ds7.classes)
    auroc_metric_central.reset()

    # auroc_metric_fl = AUROC(task="multiclass", num_classes=n_classes, average=None)
    # auroc_values_fl = auroc_metric_fl(torch.tensor(cls_fl_probs), fl.labels.cpu())
    # plot_auc(colors, fl.labels.cpu(), cls_fl_probs, auroc_values_fl, line_style='-', classes=ds7.classes)
    # auroc_metric_fl.reset()

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'')
    plt.legend(loc="lower right")
    plt.savefig(f"./auc_plots/{ds7.dataset_name}_{fl_ex}{da}.png")
    # plt.show()
