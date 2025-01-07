import copy
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from dotenv import load_dotenv
import lightning.pytorch as pl
from sklearn.preprocessing import label_binarize
import matplotlib.colors as mcolors
from dataset.datamodule_handler import get_data_modules
from models.cls import ClsMIM
from util.get_models import get_mim_model, get_cls_model
from transforms.apply_transforms import get_finetune_transformation, get_test_transformation
from util.data_labels import get_full_classes, get_merged_classes
from util.utils import set_seed
from torchmetrics import AUROC
import torch.nn.functional as F
import torch





def plot_auc(colors, labels, preds, auroc_values, line_style, classes):
    for i, color in zip(range(n_classes), colors):
        # Compute ROC curve using sklearn's roc_curve
        fpr, tpr, _ = roc_curve(labels == i, preds[:, i])
        roc_auc = auroc_values[i].item()  # Get the AUROC value for the i-th class
        if not np.any(fpr) or not np.any(tpr) or roc_auc == 0:
            continue
        # Format roc_auc value to remove trailing zeros
        roc_auc_str = f'{roc_auc:.4f}'.rstrip('0').rstrip('.')
        plt.plot(fpr, tpr, color=color, lw=2, linestyle=line_style,
                 label=f'{list(classes.keys())[i]} (area = {roc_auc_str})')


if __name__ == "__main__":
    set_seed(42)
    load_dotenv(dotenv_path="../data/.env")
    comment_1 = "mim_56"
    comment_2 = "mim_37"
    batch_size = 128
    cls_batch_size = 32
    img_size_1 = 128
    img_size_2 = 224
    epochs = 100
    devices = [1]
    mim_1 = get_mim_model(comment_1, epochs=epochs)
    mim_2 = get_mim_model(comment_2, epochs=epochs)
    param = {
        "wd": 1e-6,
        "lr": 3e-5,
        "beta1": 0.9,
        "beta2": 0.999,
    }
    data_modules_1 = get_data_modules(batch_size=cls_batch_size,
                                        classes=get_merged_classes(),
                                        train_transform=get_finetune_transformation(img_size_1),
                                        test_transform=get_test_transformation(img_size_1),
                                        filter_img=True,
                                        merge={"AMD":["CNV"], "OTHERS":["RVO", "CSC"]},
                                        DS3_3mm=True,
                                        pretraining=False)
    
    data_modules_2 = get_data_modules(batch_size=cls_batch_size,
                                        classes=get_merged_classes(),
                                        train_transform=get_finetune_transformation(img_size_2),
                                        test_transform=get_test_transformation(img_size_2),
                                        filter_img=True,
                                        merge={"AMD":["CNV"], "OTHERS":["RVO", "CSC"]},
                                        DS3_3mm=True,
                                        pretraining=False)

    for (data_module_1 ,data_module_2) in zip(data_modules_1, data_modules_2):
        # if client_name_1 == "DS1"  or client_name_1 == "DS2"  or  client_name_1 == "DS3":
        #     continue
        if data_module_1.dataset_name == "DS7":
            break
        param["step_size"] = len(data_module_1.train_dataloader())
        cls_1 = get_cls_model(f"{comment_1}_cls", copy.deepcopy(mim_1.model.encoder), client_name=data_module_1.dataset_name,
                              classes=data_module_1.classes, param=param)
        cls_2 = get_cls_model(f"{comment_2}_cls", copy.deepcopy(mim_2.model.encoder), client_name=data_module_2.dataset_name,
                              classes=data_module_2.classes, param=param)
        j = 0
        for (module_1, module_2) in zip(data_modules_1, data_modules_2):
            if module_1.dataset_name == "DS7":
                continue
            trainer = pl.Trainer(
                accelerator='gpu',
                devices=devices,
            )
            test_loader_1 = module_1.filtered_test_dataloader(data_module_1.classes)
            test_loader_2 = module_2.filtered_test_dataloader(data_module_2.classes)
            cls_1.test_classes = module_1.filtered_classes
            cls_2.test_classes = module_2.filtered_classes
            if len(cls_1.test_classes) == 0 or len(cls_2.test_classes) == 0:  # it is not possible to test it!
                continue
            test_results_1 = trainer.test(cls_1, dataloaders=test_loader_1, verbose=False)
            test_results_2 = trainer.test(cls_2, dataloaders=test_loader_2, verbose=False)

            # Convert logits to probabilities
            cls_1_probs = F.softmax(cls_1.preds, dim=1).cpu().numpy()
            cls_2_probs = F.softmax(cls_2.preds, dim=1).cpu().numpy()

            n_classes = len(data_module_1.classes)
            # Generate adaptive colors based on the number of classes
            colors = list(mcolors.TABLEAU_COLORS.values())
            plt.rcParams['font.family'] = 'Thoma'
            plt.rcParams['font.size'] = 15
            # Plot all ROC curves with predefined values from torchmetrics
            plt.figure(figsize=(8, 6))
            # Initialize the AUROC metric
            auroc_metric_1 = AUROC(task="multiclass", num_classes=n_classes, average=None)
            # Compute AUROC for each class
            auroc_values_1 = auroc_metric_1(torch.tensor(cls_1_probs), cls_1.labels.cpu())
            plot_auc(colors, cls_1.labels.cpu(), cls_1_probs, auroc_values_1, line_style='-', classes=data_module_1.classes)
            auroc_metric_1.reset()

            auroc_metric_2 = AUROC(task="multiclass", num_classes=n_classes, average=None)
            auroc_values_2 = auroc_metric_2(torch.tensor(cls_2_probs), cls_2.labels.cpu())
            plot_auc(colors, cls_2.labels.cpu(), cls_2_probs, auroc_values_2, line_style='--', classes=data_module_2.classes)
            auroc_metric_2.reset()

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'')
            # Get the number of legend elements
            legend_elements = plt.gca().get_legend_handles_labels()[1]
            # Set font size based on number of elements
            if len(legend_elements) > 8:
                plt.legend(fontsize='8', loc="lower right")
            elif len(legend_elements) > 6:
                plt.legend(fontsize='12', loc="lower right")
            else:
                plt.legend(fontsize='12', loc="lower right")
            plt.savefig(f"./auc_plots/{data_module_1.dataset_name}_{module_1.dataset_name}_128vs224.png")
            # plt.show()
            j += 1
