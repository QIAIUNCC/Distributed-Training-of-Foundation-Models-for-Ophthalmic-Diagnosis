import copy
import glob

import numpy as np
import torch
from dotenv import load_dotenv
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from dataset.OCT_dataset import OCTDataset
from dataset.datamodule_handler import get_data_modules
from local import evaluate_model
from models.cls import ClsMIM
from models.mim import MIMWrapper
from transforms.apply_transforms import get_test_transformation, get_finetune_transformation, get_pretrain_transformation
from transforms.masking import MIMTransform
from util.data_labels import get_full_classes, get_merged_classes
from util.utils import set_seed

if __name__ == "__main__":
    set_seed(42)
    load_dotenv(dotenv_path="../data/.env")
    batch_size = 32
    cls_batch_size = 32
    img_size = 128
    epochs = 100
    devices = [3]
    resume = True
    mask_type = "center"
    weights = [1]
    mask_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    patch_sizes = [4, 8, 16, 32]
    counter = 1
    for mask_ratio in mask_ratios:
        for patch_size in patch_sizes:
            architecture = f"mim_{counter}"
            bins = [(mask_ratio, mask_ratio)]
            pre_training_transform = get_pretrain_transformation(img_size=img_size)
            mim_transform = MIMTransform(img_size=img_size,
                                            model_patch_size=4,
                                            mask_patch_size=patch_size,
                                            masking_type=mask_type,
                                            weights=weights,
                                            bins=bins,
                                            transform=pre_training_transform
                                            )
            
            masked_data_modules = get_data_modules(batch_size=batch_size,
                                                classes=get_full_classes(),
                                                filter_img=False,
                                                DS3_3mm=True,
                                                pretraining=False)
            # Merging train lists
            combined_train_list = (
                    masked_data_modules[0].data_train.img_paths +
                    masked_data_modules[1].data_train.img_paths +
                    masked_data_modules[2].data_train.img_paths +
                    masked_data_modules[3].data_train.img_paths +
                    masked_data_modules[4].data_train.img_paths +
                    masked_data_modules[5].data_train.img_paths +
                    masked_data_modules[0].data_val.img_paths +
                    masked_data_modules[1].data_val.img_paths +
                    masked_data_modules[2].data_val.img_paths +
                    masked_data_modules[3].data_val.img_paths +
                    masked_data_modules[4].data_val.img_paths +
                    masked_data_modules[5].data_val.img_paths 
            )
            combined_train_dataset = OCTDataset(transform=mim_transform,
                                                data_dir="",
                                                img_paths=combined_train_list)
            train_masked_dataloader = DataLoader(combined_train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=torch.cuda.device_count() * 2,
                                                pin_memory=True
                                                )

            config = {"batch_size": batch_size, "epochs": epochs, "current_round": 1}
            # mim training
            mim = MIMWrapper(lr=2e-4,
                            wd=0.05,
                            min_lr=1e-5,
                            epochs=epochs,
                            warmup_lr=1e-6,
                            warmup_epochs=10,
                            w_ssim=0.1,
                            device="cuda:" + str(devices[0]),
                            weights=True,
                            encoder="tiny"
                            )
            save_path = os.path.join(f"checkpoints", f"{architecture}")

            tb_logger = TensorBoardLogger(save_dir=os.path.join("checkpoints", f"{architecture}"),
                                        name="centralized_mim")
            checkpoint = ModelCheckpoint(dirpath=save_path,
                                        filename="mim-{epoch}-{train_loss:.4f}", save_weights_only=False,
                                        mode="min", monitor="train_loss", save_top_k=1, save_last=True)

            trainer = pl.Trainer(accelerator='gpu', devices=devices, max_epochs=epochs,
                                callbacks=[checkpoint],
                                logger=[tb_logger],
                                log_every_n_steps=1,
                                gradient_clip_algorithm="norm",
                                gradient_clip_val=1.0,
                                deterministic=True,
                                strategy=DDPStrategy(process_group_backend='gloo', find_unused_parameters=True),
                                )
            model_path = glob.glob(os.path.join(save_path, f"mim*.ckpt"))
            if len(model_path) > 0:
                if resume:
                    trainer.fit(model=mim,
                                ckpt_path=model_path[0],
                                train_dataloaders=train_masked_dataloader)
                else:
                    mim = MIMWrapper.load_from_checkpoint(model_path[0],
                                                        lr=2e-4,
                                                        wd=0.05,
                                                        min_lr=1e-5,
                                                        epochs=epochs,
                                                        warmup_lr=1e-6,
                                                        warmup_epochs=10,
                                                        gamma=0.1,
                                                        device="cuda:" + str(devices[0]),
                                                        weights=True,
                                                        encoder="tiny"
                                                        )

            else:
                trainer.fit(model=mim,
                            train_dataloaders=train_masked_dataloader)
            counter += 1
            ### classification models
            param = {
                "wd": 1e-6,
                "lr": 3e-5,
                "beta1": 0.9,
                "beta2": 0.999,
            }
            cls_data_modules = get_data_modules(batch_size=cls_batch_size,
                                                classes=get_merged_classes(),
                                                train_transform=get_finetune_transformation(img_size),
                                                test_transform=get_test_transformation(img_size,),
                                                filter_img=True,
                                                merge={"AMD":["CNV"], "OTHERS":["RVO", "CSC"]},
                                                DS3_3mm=True)[:-5]
            save_path = os.path.join("checkpoints", f"{architecture}_cls")
            config = {"batch_size": cls_batch_size, "epochs": epochs, "current_round": 1}

            for data_module in cls_data_modules:
                param["step_size"] = len(data_module.train_dataloader())
                model_path = glob.glob(
                    os.path.join(save_path, data_module.dataset_name, "version_0", "checkpoints", f"mim_*.ckpt"))
                if len(model_path) > 0:
                    model = ClsMIM.load_from_checkpoint(model_path[0],
                                                        encoder=copy.deepcopy(mim.model.encoder),
                                                        wd=param["wd"],
                                                        lr=param["lr"],
                                                        beta1=param["beta1"],
                                                        beta2=param["beta2"],
                                                        step_size=param["step_size"],
                                                        gamma=0.5,
                                                        classes=data_module.classes,
                                                        feature_dim=1000
                                )
                else:
                    model = ClsMIM(encoder=copy.deepcopy(mim.model.encoder),
                                wd=param["wd"],
                                lr=param["lr"],
                                beta1=param["beta1"],
                                beta2=param["beta2"],
                                step_size=param["step_size"],
                                gamma=0.5,
                                classes=data_module.classes,
                                feature_dim=1000
                                )
                early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False,
                                            mode="min")
                tb_logger = TensorBoardLogger(save_dir=save_path, name=data_module.dataset_name)
                trainer = pl.Trainer(accelerator='gpu', devices=[1], max_epochs=epochs,
                                    callbacks=[early_stopping],
                                    logger=[tb_logger],
                                    log_every_n_steps=1,
                                    deterministic=True,
                                    )


                trainer.fit(model=model, datamodule=data_module)
                evaluate_model(model=model,
                            client_name=data_module.dataset_name,
                            data_modules=cls_data_modules,
                            approach=f"centralized_{mask_type}",
                            config=config,
                            classes=data_module.classes,
                            architecture=architecture,
                            devices=devices)
