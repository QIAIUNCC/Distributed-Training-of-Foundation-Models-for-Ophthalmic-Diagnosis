# Compute and plot the macro-average ROC curve for each model
import copy

from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.preprocessing import label_binarize

from dataset.data_module_handler import get_data_modules
from plots.auc_plots_DS7 import get_mim_model, get_cls_model, get_local_model
from transforms.apply_transforms import get_test_transformation, get_finetune_transformation
from util.data_labels import get_merged_classes
from util.utils import set_seed


import lightning.pytorch as pl
import torch.nn.functional as F


def plot_macro_avg_roc(model_name, probs, labels, color, linestyle):
    # Binarize labels for multiclass ROC
    labels_binarized = label_binarize(labels, classes=list(range(len(ds7.classes))))

    # Compute ROC curve and AUC for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(ds7.classes)):
        fpr[i], tpr[i], _ = roc_curve(labels_binarized[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    avg_auc = np.mean(list(roc_auc.values()))  # Compute macro-average AUC

    # Calculate the macro-average ROC curve by averaging TPRs at unique FPR points
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(ds7.classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(ds7.classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(ds7.classes)

    # Plot the macro-average ROC curve for this model
    plt.plot(all_fpr, mean_tpr, color=color, lw=2, linestyle=linestyle,
             label=f'{model_name} (Macro AUC = {avg_auc:.4f})')


if __name__ == "__main__":
    set_seed(42)
    load_dotenv(dotenv_path="../data/.env")

    # Initialize models and datasets (as in your existing code)
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
                           env_path="../data3/.env")[-3]

    ds7_da = get_data_modules(batch_size=cls_batch_size,
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

    # Load models
    local = get_local_model(architecture=architecture_local, classes=ds7.classes, client_name=ds7.dataset_name)
    mim_c = get_mim_model(architecture_mim, epochs=epochs)
    central = get_cls_model(architecture_central, copy.deepcopy(mim_c.model.encoder), ds7.classes, ds7.dataset_name,
                            param)

    # Load federated models with different configurations
    fl_ex_1 = "14_f_1"
    architecture_fl_1 = f"fl_ex{fl_ex_1}_100_0.5"
    fl_1 = get_cls_model(architecture_fl_1, copy.deepcopy(mim_c.model.encoder), ds7_da.classes, ds7_da.dataset_name, param)

    fl_ex_5 = "14_f_5"
    architecture_fl_5 = f"fl_ex{fl_ex_5}_100_0.5"
    fl_5 = get_cls_model(architecture_fl_5, copy.deepcopy(mim_c.model.encoder), ds7_da.classes, ds7_da.dataset_name, param)

    trainer = pl.Trainer(accelerator='gpu', devices=devices, log_every_n_steps=1, deterministic=True)

    test_loader = ds7.filtered_test_dataloader(ds7.classes)
    test_results_local = trainer.test(local, dataloaders=ds7.test_dataloader(), verbose=False)
    test_results_central = trainer.test(central, dataloaders=ds7.test_dataloader(), verbose=False)
    test_results_fl_1 = trainer.test(fl_1, dataloaders=ds7_da.test_dataloader(), verbose=False)
    test_results_fl_5 = trainer.test(fl_5, dataloaders=ds7_da.test_dataloader(), verbose=False)

    # Obtain test predictions and convert logits to probabilities
    cls_local_probs = F.softmax(local.preds, dim=1).cpu().numpy()
    cls_central_probs = F.softmax(central.preds, dim=1).cpu().numpy()
    cls_fl_1_probs = F.softmax(fl_1.preds, dim=1).cpu().numpy()
    cls_fl_5_probs = F.softmax(fl_5.preds, dim=1).cpu().numpy()

    plt.rcParams['font.family'] = 'Thoma'
    plt.rcParams['font.size'] = 17
    # Plot macro-average ROC curves
    plt.figure(figsize=(8, 6))
    plot_macro_avg_roc("Local", cls_local_probs, local.labels.cpu(), color="blue", linestyle=':')
    plot_macro_avg_roc("Central", cls_central_probs, central.labels.cpu(), color="orange", linestyle='-.')
    plot_macro_avg_roc("DAD-FDL-1", cls_fl_1_probs, fl_1.labels.cpu(), color="green", linestyle='--')
    plot_macro_avg_roc("DAD-FDL-5", cls_fl_5_probs, fl_5.labels.cpu(), color="red", linestyle='-')

    # Plot baseline and configure plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('')
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save and display the plot
    plt.savefig(f"./auc_plots/macro_avg_auc_comparison_{ds7.dataset_name}.png")
    plt.show()
