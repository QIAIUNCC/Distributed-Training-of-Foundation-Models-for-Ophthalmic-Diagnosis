import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseNet, FocalLoss
import torch



class MLP(nn.Module):
    def __init__(self, in_features, num_classes, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=in_features * 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(in_features=in_features * 2, out_features=in_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.dense1(x)))
        x = self.dropout2(self.relu2(self.dense2(x)))
        x = self.output_layer(x)
        return x


class ClsMIM(BaseNet):
    def __init__(self, feature_dim, **kwargs):
        super().__init__(**kwargs)
        if feature_dim != 0:
            self.encoder = nn.Sequential(kwargs["encoder"],
                                         MLP(feature_dim, len(kwargs["classes"]))
                                         )
        else:
            self.encoder = kwargs["encoder"]

        self.criterion = FocalLoss()

    def _calculate_loss(self, batch):
        imgs, labels = batch
        preds = self.forward(imgs)
        loss = self.criterion(preds, labels)
        return {"loss": loss, "preds": preds, "labels": labels, "pred_max": preds.argmax(dim=-1)}