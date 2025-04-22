import os

from datamodule import MultistepDataModule
from modelmodule import MultistepForecastModule
from pytorch_lightning.cli import LightningCLI


def train(datamodule, modelmodule):

    print("Loading data from", os.environ['DATA_DIR'])
    print("Saving results to", os.environ['RESULTS_DIR'])

    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=modelmodule,
        datamodule_class=datamodule,
        # Overwrite the existing config file
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )

    print("Lightning Seed", os.environ['PL_GLOBAL_SEED'])

    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    for logger in cli.trainer.loggers:
        logger.log_hyperparams(cli.config.as_dict())

    # Models need access to the datamodule
    cli.model.set_datamodule(cli.datamodule)

    print('\nStart training\n')
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    print('\nStart testing\n')
    if cli.trainer.fast_dev_run == False and cli.trainer.overfit_batches == 0:
        # test the trained model
        cli.trainer.test(cli.model, datamodule=cli.datamodule,
                         ckpt_path="best")


if __name__ == "__main__":
    train(MultistepDataModule, MultistepForecastModule)
