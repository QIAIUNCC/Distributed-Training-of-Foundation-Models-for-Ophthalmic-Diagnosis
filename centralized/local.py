import glob
import torchvision.models
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torchvision.models import Swin_V2_T_Weights, Swin_V2_B_Weights

from dataset.datamodule_handler import get_data_modules
from models.base import BaseNet
from models.cls import MLP
from models.evaluate_model import evaluate_model
from transforms.apply_transforms import get_finetune_transformation, get_test_transformation, get_finetune_transformation
from util.data_labels import get_merged_classes
from util.utils import set_seed
from util.get_models import get_baseline_model

if __name__ == "__main__":
    set_seed(42)
    comment = "local_5"
    model_architecture = "resnet50"
    pretrained = True
    resume = False
    batch_size = 128
    img_size = 128
    devices = [0]
    epochs = 100

    config = {"batch_size": batch_size, "epochs": epochs, "current_round": 1}
    data_modules = get_data_modules(batch_size=batch_size,
                                        classes=get_merged_classes(),
                                        train_transform=get_finetune_transformation(img_size),
                                        test_transform=get_test_transformation(img_size,),
                                        filter_img=True,
                                        merge={"AMD":["CNV"], "OTHERS":["RVO", "CSC"]},
                                        DS3_3mm=True,
                                        pretraining=False,
                                        env_path = "../data/.env")
    for data_module in data_modules:
        save_path = os.path.join(f"./checkpoints", comment, data_module.dataset_name)

        early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False,
                                       mode="min")
        tb_logger = TensorBoardLogger(save_dir=save_path, name="tb")
        checkpoint = ModelCheckpoint(dirpath=save_path,
                                     filename=model_architecture + "_-{epoch}-{val_auc:.4f}",
                                     mode="min", monitor="val_loss", save_top_k=1)

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=devices,
            max_epochs=config["epochs"],
            callbacks=[early_stopping, checkpoint],
            logger=[tb_logger],
            log_every_n_steps=1
        )

        if os.path.exists(f"log_{epochs}/{data_module.dataset_name}_{comment}.txt"):
            print(f"{data_module.dataset_name} {model_architecture} skipped!")
            continue
        # encoder = get_swingv2_model(pretrained=pretrained, model=swinv2_model)
        encoder = get_baseline_model(pretrained=pretrained, model_architecture=model_architecture)
        encoder = nn.Sequential(encoder, MLP(1000, len(data_module.classes)))
        # encoder = torchvision.models.swin_v2_t(num_classes=len(classes))
        model_path = glob.glob(os.path.join(save_path, f"{model_architecture}_*.ckpt"))
        if len(model_path) > 0:
            model = BaseNet.load_from_checkpoint(model_path[0], classes=data_module.classes, lr=3e-5, wd=1e-6, encoder=encoder, map_location = {'cuda:2':'cuda:1'})
        else:
            model = BaseNet(classes=data_module.classes, lr=3e-5, wd=1e-6, encoder=encoder)
            # train the model
            trainer.fit(model=model, datamodule=data_module)
        model_path = glob.glob(os.path.join(save_path, f"{model_architecture}_*.ckpt"))
        model = BaseNet.load_from_checkpoint(model_path[0], classes=data_module.classes, lr=3e-5, wd=1e-6, encoder=encoder,map_location = {'cuda:2':'cuda:1'})
        # Test the model
        evaluate_model(model=model,
                       client_name=data_module.dataset_name,
                       data_modules=data_modules,
                       comment=comment,
                       config=config,
                       classes=data_module.classes,
                       devices=devices,
                       approach="centralized")
