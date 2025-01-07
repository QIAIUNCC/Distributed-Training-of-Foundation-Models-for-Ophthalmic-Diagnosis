import copy
import glob
import torch
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from dataset.datamodule_handler import get_data_modules
from models.cls import ClsMIM
from models.evaluate_model import evaluate_model
from models.mim import MIMWrapper
from transforms.apply_transforms import get_test_transformation, get_finetune_transformation
from util.data_labels import get_merged_classes
from util.utils import set_seed

if __name__ == "__main__":
    set_seed(42)
    architecture = "fl_ex15_5"
    batch_size = 128
    cls_batch_size = 32
    img_size = 128
    epochs = 100
    devices = [1]
    mask_ratio = 0.5
    cls_data_modules = get_data_modules(batch_size=cls_batch_size,
                                        classes=get_merged_classes(),
                                        train_transform=get_finetune_transformation(img_size,
                                                                                    apply_adaptation=False),
                                        test_transform=get_test_transformation(img_size,
                                                                               apply_adaptation=False),
                                        env_path="../data/.env"

                                        )
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
    state_dict = torch.load('../fl/ex15_f_5.pth')
    save_path = os.path.join(f"checkpoints", f"{architecture}_{epochs}_{mask_ratio}")
    # Adjust the keys by removing the 'model.' prefix
    adjusted_state_dict = {'model.' + key: value for key, value in state_dict.items()}
    mim.load_state_dict(adjusted_state_dict)
    ### classification models
    param = {
        "wd": 1e-6,
        "lr": 3e-5,
        "beta1": 0.9,
        "beta2": 0.999,
    }
    config = {"batch_size": cls_batch_size, "epochs": epochs, "current_round": 1}
    data_module = cls_data_modules[-3]
    param["step_size"] = len(data_module.train_dataloader())
    model_path = glob.glob(os.path.join(save_path, data_module.dataset_name, "version_0", "checkpoints", "*.ckpt"))
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
        trainer = pl.Trainer(accelerator='gpu', devices=devices, max_epochs=epochs,
                             callbacks=[early_stopping],
                             logger=[tb_logger],
                             log_every_n_steps=1,
                             deterministic=True,
                             )
        trainer.fit(model=model, datamodule=data_module)
    evaluate_model(model=model,
                   client_name=data_module.dataset_name,
                   data_modules=cls_data_modules,
                   approach=f"centralized_{int(mask_ratio * 10)}",
                   config=config,
                   classes=data_module.classes,
                   architecture=architecture,
                   devices=devices)
