import lightning.pytorch as pl
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import F1Score, AUROC, PrecisionRecallCurve
from torchmetrics.classification import MulticlassPrecision, MulticlassF1Score, MulticlassAUROC, MulticlassPrecisionRecallCurve
from torchmetrics.classification.accuracy import MulticlassAccuracy

from sklearn.metrics import precision_recall_curve, auc


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # This assumes that 'inputs' is the output of a softmax or log_softmax layer,
        # and 'targets' is a batch of ground-truth class indices.
        # If using log_softmax, you should apply torch.exp to 'inputs' to get the probabilities.
        log_prob = F.log_softmax(inputs, dim=-1)
        prob = torch.exp(log_prob)

        # Gather the probabilities of the targeted classes
        targets_prob = prob.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Calculate the focal loss
        focal_loss = -self.alpha * ((1 - targets_prob) ** self.gamma) * log_prob.gather(dim=-1, index=targets.unsqueeze(
            -1)).squeeze(-1)

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class BaseNet(pl.LightningModule):
    def __init__(self, classes, lr, encoder, wd=0, momentum=0, dampening=0, optimizer="AdamW", beta1=0.9, beta2=0.999,
                 step_size=1000, gamma=0.5):
        """

        :param classes (tuple(str, int)): list of tuples, each tuple consists of class name and class index
        :param lr (float): learning rate
        :param weight_decay (float): weight decay of optimizer
        """
        super().__init__()
        task = "binary" if len(classes) == 2 else "multiclass"
        self.metrics_list = ["accuracy", "precision", "f1", "auc", "pr"]
        self.sessions = ["train", "val", "test"]

        self.train_ac = MulticlassAccuracy(num_classes=len(classes), average=None)
        self.val_ac = MulticlassAccuracy(num_classes=len(classes), average=None)
        self.test_ac = MulticlassAccuracy(num_classes=len(classes), average=None)

        self.train_p = MulticlassPrecision(num_classes=len(classes), average=None)
        self.val_p = MulticlassPrecision(num_classes=len(classes), average=None)
        self.test_p = MulticlassPrecision(num_classes=len(classes), average=None)

        self.train_f1 = MulticlassF1Score(num_classes=len(classes), average=None)
        self.val_f1 = MulticlassF1Score(num_classes=len(classes), average=None)
        self.test_f1 = MulticlassF1Score(num_classes=len(classes), average=None)

        self.train_auc = MulticlassAUROC(num_classes=len(classes), average=None)
        self.val_auc = MulticlassAUROC(num_classes=len(classes), average=None)
        self.test_auc = MulticlassAUROC(num_classes=len(classes), average=None)

        self.train_pr = MulticlassPrecisionRecallCurve(num_classes=len(classes))
        self.val_pr = MulticlassPrecisionRecallCurve(num_classes=len(classes))
        self.test_pr = MulticlassPrecisionRecallCurve( num_classes=len(classes))

        self.metrics = {"train": [self.train_ac, self.train_p, self.train_f1, self.train_auc, self.train_pr],
                        "val": [self.val_ac, self.val_p, self.val_f1, self.val_auc, self.val_pr],
                        "test": [self.test_ac, self.test_p, self.test_f1, self.test_auc, self.test_pr],
                        }

        self.step_output = {"train": [], "val": [], "test": []}
        self.mean_log_keys = ["loss"]
        self.lr = lr
        self.wd = wd
        self.momentum = momentum
        self.dampening = dampening
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.step_size = step_size
        self.gamma = gamma
        self.classes = classes
        self.encoder = encoder
        self.test_classes = classes

    def configure_optimizers(self):
        optimizer = None
        if self.optimizer == "AdamW":
            optimizer = optim.AdamW(self.encoder.parameters(), lr=self.lr, weight_decay=self.wd,
                                    betas=(self.beta1, self.beta2))
        elif self.optimizer == "Adam":
            optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == "SGD":
            optimizer = optim.SGD(self.encoder.parameters(), lr=self.lr, weight_decay=self.wd,
                                  dampening=self.dampening, momentum=self.momentum)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def forward(self, x):
        return self.encoder(x)

    def _calculate_loss(self, batch):
        imgs, labels = batch
        preds = self.forward(imgs)
        loss = F.cross_entropy(preds, labels)
        return {"loss": loss, "preds": preds, "labels": labels}


    def training_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.step_output["train"].append(output)
        return output

    def on_train_epoch_end(self):
        self.stack_update(session="train")

    def validation_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.step_output["val"].append(output)
        return output

    def on_validation_epoch_end(self):
        self.stack_update(session="val")

    def test_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.step_output["test"].append(output)
        return output

    def on_test_epoch_end(self, ):
        self.stack_update(session="test")

    def update_metrics(self, session, preds, labels):
        for metric in self.metrics[session]:
            metric.update(preds, labels)

    def stack_update(self, session):
        all_preds = torch.cat([out["preds"] for out in self.step_output[session]])
        all_labels = torch.cat([out["labels"] for out in self.step_output[session]])
        log = {}
        for key in self.mean_log_keys:
            log[f"{session}_{key}"] = torch.stack([out[key] for out in self.step_output[session]]).mean()

        self.update_metrics(session=session, preds=all_preds, labels=all_labels)
        res = self.compute_metrics(session=session)
        self.add_log(session, res, log)
        if session == "test":
            # _, self.preds = torch.max(all_preds, 1)
            self.preds = all_preds
            self.labels = all_labels
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)
        self.restart_metrics(session=session)

        return all_preds, all_labels

    def compute_metrics(self, session):
        res = {}
        for metric, metric_name in zip(self.metrics[session], self.metrics_list):
            res[metric_name] = metric.compute()
        return res

    def restart_metrics(self, session):
        for metric in self.metrics[session]:
            metric.reset()
        self.step_output[session].clear()  # free memory

    def add_log(self, session, res, log):
        for metric in self.metrics_list:
            log[session + "_" + metric] = 0
        if "pr" in self.metrics_list:
            precision, recall, _ = res["pr"]
            indices = torch.tensor(list(sorted(self.test_classes.values())))
            # Calculating area under the curve for each class
            aucs = []
            for i in indices:
                # Sort the recall and precision arrays
                recall_sorted_indices = recall[i].argsort()
                sorted_recall = recall[i][recall_sorted_indices]
                sorted_precision = precision[i][recall_sorted_indices]

                # Calculate AUC using sklearn's auc function
                class_auc = auc(sorted_recall.cpu().numpy(), sorted_precision.cpu().numpy())
                aucs.append(class_auc)

            # Average AUC across all classes
            average_auc = sum(aucs) / len(aucs)
            log[session + "_pr"] = average_auc
            # log[session + "_pr"] = res["pr"]
        for idx, key in enumerate(self.test_classes.keys()):
            log[session + "_precision_" + key] = res["precision"][idx]
            log[session + "_precision"] += res["precision"][idx]
            log[session + "_accuracy_" + key] = res["accuracy"][idx]
            log[session + "_accuracy"] += res["accuracy"][idx]
            log[session + "_auc_" + key] = res["auc"][idx]
            log[session + "_auc"] += res["auc"][idx]
            log[session + "_f1_" + key] = res["f1"][idx]
            log[session + "_f1"] += res["f1"][idx]
        log[session + "_f1"] /= len(self.test_classes)
        log[session + "_accuracy"] /= len(self.test_classes)
        log[session + "_precision"] /= len(self.test_classes)
        log[session + "_auc"] /= len(self.test_classes)
