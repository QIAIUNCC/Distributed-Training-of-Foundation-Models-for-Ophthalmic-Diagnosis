import os
import flwr as fl
import pytorch_lightning as pl
from collections import OrderedDict
import torch
from pytorch_lightning.callbacks import EarlyStopping

from util.log_handler import log_results
from util.utils import set_seed


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, client_name, experiment_name):
        self.net = net
        self.client_name = client_name,
        self.experiment_name = experiment_name

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
        set_seed()
        self.set_parameters(parameters, config)
        early_stopping = EarlyStopping(monitor=config["monitor"], patience=config["patience"], verbose=False,
                                       mode=config["mode"])
        trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=config["max_epochs"],
                             callbacks=[early_stopping],
                             logger=False,
                             enable_checkpointing=False,
                             # log_every_n_steps=config["log_n_steps"],
                             )
        trainer.fit(model=self.net, datamodule=self.datamodule)

        return self.get_parameters(self.net.model), len(self.datamodule.train_dataloader()), {}

    def evaluate(self, parameters, config):
        """
        Receive model parameters from the server, evaluate the model parameters on the local data,
        and return the evaluation result to the server
        :param parameters:
        :param config:
        :return:
        """
        self.set_parameters(parameters, config)
        trainer = pl.Trainer(accelerator='gpu', devices=[0], log_every_n_steps=1)
        print("============================")
        test_results = trainer.test(self.net, self.datamodule.testdata_loader(), verbose=True)

        loss = test_results[0]["test_loss"]

        log_results(classes=self.net.hparams.classes,
                    results=test_results,
                    client_name=self.client_name,
                    comment=self.experiment_name,
                    config=config)
        print("================", self.client_name, "==============")
        print("f1:", test_results[0]["test_f1"])
        print("auc:", test_results[0]["test_auc"])
        print("loss:", test_results[0]["test_loss"])

        return float(loss), len(self.datamodule.test_loader()), test_results[0]


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    # pass
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


