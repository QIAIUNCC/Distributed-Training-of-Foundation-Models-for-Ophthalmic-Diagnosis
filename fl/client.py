import copy
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import flwr as fl
from centralized.local import evaluate_model
from dataset.OCT_dataset import OCTDataset
from dataset.datamodule_handler import get_data_modules, get_datamodule
import os
from fl.lib.fl_client import _get_parameters, _set_parameters
from models.cls import ClsMIM
from models.mim import MIMWrapper
from transforms.apply_transforms import get_test_transformation, get_finetune_transformation, \
    get_pretrain_transformation
from transforms.masking import MIMTransform
from util.data_labels import get_merged_classes, get_full_classes
from util.log_handler import read_val_results, is_null_like, write_val_results
from util.utils import set_seed


def get_mask_data(img_size, dataset_name, dataset_path, batch_size, apply_adaptation=True, dad_img_path=""):
    classes = get_full_classes()
    bins = [(0.5, 0.5)]
    pre_training_transform = get_pretrain_transformation(img_size=img_size, dad_img_path=dad_img_path)
    mim_transform = MIMTransform(img_size=img_size,
                                    model_patch_size=4,
                                    mask_patch_size=4,
                                    masking_type="center",
                                    bins=bins,
                                    weights=[1],
                                    transform=pre_training_transform,
                                    )
    masked_datamodule = get_datamodule(dataset_name=dataset_name,
                                       dataset_path=dataset_path,
                                       batch_size=batch_size,
                                       kermany_classes=classes[0],
                                       srinivasan_classes=classes[1],
                                       oct500_classes=classes[2],
                                       nur_classes=classes[3],
                                       waterloo_classes=classes[4],
                                       octdl_classes=classes[5],
                                       train_transform=mim_transform,
                                       test_transform=get_test_transformation(img_size,
                                                                              apply_adaptation=apply_adaptation,
                                                                              dad_img_path=dad_img_path),
                                       filter_img=False
                                       )

    train_dataset = OCTDataset(transform=mim_transform,
                               data_dir="",
                               img_paths=masked_datamodule.data_train.img_paths)
    val_dataset = OCTDataset(transform=mim_transform,
                             data_dir="",
                             img_paths=masked_datamodule.data_val.img_paths)
    train_mask_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=True,
                                   num_workers=torch.cuda.device_count() * 2,
                                   )
    val_mask_loader = DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 num_workers=torch.cuda.device_count() * 2,
                                 pin_memory=True,
                                 )

    return train_mask_loader, val_mask_loader


class FlowerClientMim(fl.client.NumPyClient):
    def __init__(self, net, cid):
        self.net = net
        self.client_name = "DS" + str((int(cid) + 1))
        self.cid = cid
        load_dotenv(dotenv_path="./data/.env")
        self.dataset_path = os.getenv(self.client_name + "_PATH")

    def get_parameters(self, config):
        """
        Return the current local model parameters
        :param config:
        :return:
        """
        return _get_parameters(self.net.model)

    def set_parameters(self, parameters, config):
        _set_parameters(self.net.model, parameters)

    def fit(self, parameters, config):
        """
        Receive model parameters from the server, train the model parameters on the local data,
        and return the (updated) model parameters to the server
        :param parameters:
        :param config: dictionary contains the fit configuration
        :return: local model's parameters, length train data,
        """
        set_seed(42)
        mask_train_loader, mask_val_loader = get_mask_data(img_size=config["img_size"],
                                                           dataset_name=self.client_name,
                                                           batch_size=config["batch_size"],
                                                           dataset_path=self.dataset_path,
                                                           apply_adaptation=True,
                                                           dad_img_path=config["dad_image_path"]
                                                           )
        self.set_parameters(parameters, config)
        early_stopping = EarlyStopping(monitor=config["monitor"],
                                       patience=config["patience"],
                                       verbose=False,
                                       mode=config["mode"])
        self.net.warmup_epoch = config["warmup_epoch"]
        trainer = pl.Trainer(accelerator='gpu',
                             devices=1,
                             max_epochs=config["epochs"],
                             callbacks=[early_stopping],
                             enable_checkpointing=False,
                             log_every_n_steps=1,
                             gradient_clip_algorithm="norm",
                             gradient_clip_val=config["gradient_clip_val"]
                             # log_every_n_steps=config["log_n_steps"],
                             )
        trainer.fit(model=self.net,
                    train_dataloaders=mask_train_loader,
                    val_dataloaders=mask_val_loader)
        return self.get_parameters(self.net.model), len(mask_val_loader), {}

    def evaluate(self, parameters, config):
        """
              Receive model parameters from the server, evaluate the model parameters on the local data,
              and return the evaluation result to the server
              :param parameters:
              :param config:
              :return:
              """
        self.set_parameters(parameters, config)
        cls_classes = get_merged_classes()
        cls_datamodule = get_datamodule(dataset_name=self.client_name,
                                        dataset_path=self.dataset_path,
                                        batch_size=config["batch_size"],
                                        kermany_classes=cls_classes[0],
                                        srinivasan_classes=cls_classes[1],
                                        oct500_classes=cls_classes[2],
                                        nur_classes=cls_classes[3],
                                        waterloo_classes=cls_classes[4],
                                        octdl_classes=cls_classes[5],
                                        train_transform=get_finetune_transformation(config["img_size"],
                                                                                    apply_adaptation=True,
                                                                                    dad_img_path=config["dad_image_path"]),
                                        test_transform=get_test_transformation(config["img_size"],
                                                                               apply_adaptation=True,
                                                                               dad_img_path=config["dad_image_path"]),
                                        merge={"AMD": ["CNV"], "OTHERS": ["RVO", "CSC"]}
                                        )

        early_stopping = EarlyStopping(monitor=config["monitor"],
                                       patience=config["patience"],
                                       verbose=False,
                                       mode=config["mode"])
        trainer = pl.Trainer(accelerator='gpu',
                             devices=1,
                             max_epochs=config["epochs"],
                             callbacks=[early_stopping],
                             enable_checkpointing=False,
                             log_every_n_steps=1
                             )
        model = ClsMIM(
            encoder=copy.deepcopy(self.net.model.encoder),
            wd=config["ft_wd"],
            lr=config["ft_lr"],
            beta1=config["ft_beta1"],
            beta2=config["ft_beta2"],
            step_size=len(cls_datamodule.train_dataloader()),
            classes=cls_classes[int(self.cid)],
            gamma=0.5,
            feature_dim=1000
        )
        # train the model
        trainer.fit(model=model,
                    train_dataloaders=cls_datamodule.train_dataloader(),
                    val_dataloaders=cls_datamodule.val_dataloader())

        val_results = trainer.test(model, dataloaders=cls_datamodule.val_dataloader(), verbose=False)
        dir_name = f"log_{config['epochs']}"
        log_name = f'{dir_name}/{self.client_name}_{config["comment"]}_val.json'
        print(log_name)
        res = read_val_results(log_name)
        print(res)
        if not res:
            write = True
        else:
            prev_score = float(res["test_f1"]) + float(res["test_auc"]) + float(res["test_pr"])
            result = {k: (0 if is_null_like(v) else v) for k, v in val_results[0].items()}
            new_score = float(result["test_f1"]) + float(result["test_auc"]) + float(result["test_pr"])
            write = True if new_score > prev_score else False
        if write:
            # evaluate on test sets
            data_modules = get_data_modules(batch_size=config["batch_size"],
                                            classes=get_merged_classes(),
                                            train_transform=get_finetune_transformation(config["img_size"],
                                                                                        apply_adaptation=True,
                                                                                        dad_img_path=config["dad_image_path"]),
                                            test_transform=get_test_transformation(config["img_size"],
                                                                                   apply_adaptation=True,
                                                                                   dad_img_path=config["dad_image_path"]),
                                                                                   pretraining=False)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            write_val_results(result=val_results, file_path=log_name)
            evaluate_model(model=model,
                           classes=cls_classes[int(self.cid)],
                           client_name=self.client_name,
                           data_modules=data_modules,
                           comment=config["comment"],
                           approach="fl",
                           config=config,
                           devices=[0])

        return float(val_results[0]["test_loss"]), len(cls_datamodule.val_dataloader()), val_results[0]


def client_fn_Mim(cid: str) -> FlowerClientMim:
    """Creates a FlowerClient instance on demand
    Create a Flower client representing a single organization
    """
    mim = MIMWrapper(lr=1.5e-3,
                          wd=0.05,
                          min_lr=1e-7,
                          patience=10,
                          epochs=100,
                          warmup_lr=1e-7,
                          warmup_epochs=10,
                          gamma=0.1,
                          weights=True
                          )
    return FlowerClientMim(net=mim,
                           cid=int(cid))

# if __name__ == "__main__":
#     set_seed(10)
#     server_ip = os.getenv('SERVER_IP')
#     trials = 10
#
#     client_name = str(os.getenv('CLIENT_NAME'))
#     param = get_hyperparameters(client_name, "ViT")
#
#     # Model and data
#     for i in range(0, trials):
#         simim = SimMimWrapper(num_classes=len(classes),
#                               lr=5e-4,
#                               warmup_lr=5e-7,
#                               wd=0.05,
#                               min_lr=5e-6,
#                               epochs=300,
#                               warmup_epochs=20,
#                               )
#
#         model = FedMim(encoder=simim.model.encoder,
#                        wd=param["wd"],
#                        lr=param["lr"],
#                        beta1=param["beta1"],
#                        beta2=param["beta2"],
#                        step_size=len(cls_train_loader * batch_size) // 2, gamma=0.5
#                        )
#
#         client = FlowerClientMim(simim, model, train_loader, val_loader, test_loader,
#                                  client_name=client_name,
#                                  architecture=architecture)
#         fl.client.start_numpy_client(server_address=f"{server_ip}:{os.getenv('SERVER_PORT')}",
#                                      client=client)
#         torch.cuda.empty_cache()
#         sleep(10)
