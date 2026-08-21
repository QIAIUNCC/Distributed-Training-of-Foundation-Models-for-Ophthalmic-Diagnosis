import os
import flwr as fl
from dotenv import load_dotenv
from client import client_fn_Mim
from fl.lib.fl_strategy import FedAvgStrategy
from models.mim import MIMWrapper
from util.utils import weighted_average, set_seed


def main(net, server_port, comment) -> None:
    # Define strategy
    strategy = FedAvgStrategy(
        net=net,
        on_fit_config_fn=fit_config,  # The fit_config function we defined earlier
        on_evaluate_config_fn=eval_config,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=num_clients,  # Never sample less than num_clients for training
        min_evaluate_clients=num_clients,  # Never sample less than num_clients for evaluation
        # # Minimum number of clients that need to be connected to the server before a training round can start
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        comment=comment
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address=f"0.0.0.0:{server_port}",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "epochs": epochs,
        "patience": patience,
        "monitor": "val_loss",
        "mode": "min",
        "clients": num_clients,
        "batch_size": batch_size,
        "img_size": img_size,
        "warmup_epoch": 5,
        "gradient_clip_val": 1,
        "dad_image_path": dad_image_path,
        "comment": comment
    }
    return config


def eval_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    config = {
        "batch_size": batch_size,
        "current_round": server_round,
        "clients": num_clients,
        "epochs": epochs,
        "patience": patience,
        "monitor": "val_loss",
        "mode": "min",
        "img_size": img_size,
        "ft_wd": 1e-6,
        "ft_lr": 3e-5,
        "ft_beta1": 0.9,
        "ft_beta2": 0.999,
        "dad_image_path": dad_image_path,
        "comment":comment,
    }
    return config


def simulation_main(net, client_fn, comment) -> None:
    # Create FedAvg strategy
    strategy = FedAvgStrategy(
        net=net,
        comment=comment,
        on_fit_config_fn=fit_config,  # The fit_config function we defined earlier
        on_evaluate_config_fn=eval_config,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1,  # Sample 50% of available clients for evaluation
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources={"num_gpus": 1, "num_cpus": (os.cpu_count()-4) // 4},
    )


if __name__ == "__main__":
    set_seed(42)
    num_clients = 6
    comment="fl_20"
    epochs = 100
    patience = 10
    img_size = 128
    batch_size = 32
    # dad_image_path = "../DS2-target_image.tif"
    # dad_image_path = "../DS3-target_image.bmp"
    # dad_image_path = "../DS4-target_image.tif"
    # dad_image_path = "../DS5-target_image.jpeg"
    dad_image_path = "../transforms/DS1-target_image.jpeg"


    load_dotenv(dotenv_path="../data/.env")
    mim = MIMWrapper(lr=5e-4,
                            warmup_lr=5e-7,
                            wd=0.05,
                            min_lr=5e-7,
                            epochs=100,
                            warmup_epochs=5,
                            weights=False
                            )
    simulation_main(net=mim, client_fn=client_fn_Mim, comment=comment)
