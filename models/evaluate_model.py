import lightning.pytorch as pl
from util.log_handler import log_results


def evaluate_model(model, classes, client_name, data_modules, comment, approach, config, devices):
    for data_module in data_modules:
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=devices,
            max_epochs=config["epochs"],
        )
        test_loader = data_module.filtered_test_dataloader(classes)
        model.test_classes = data_module.filtered_classes
        if len(model.test_classes) == 0:  #it is not possible to test it!
            continue
        test_results = trainer.test(model, dataloaders=test_loader, verbose=False)
        print(test_results)
        
        config['test_set'] = data_module.dataset_name
        log_results(classes=data_module.filtered_classes,
                    results=test_results,
                    client_name=client_name,
                    comment=comment,
                    approach=approach,
                    config=config)
