from omegaconf import OmegaConf, DictConfig
import hydra
import joblib
from pathlib import Path

import optuna
# from optuna.integration import PyTorchLightningPruningCallback
import torch
import logging
import pytorch_lightning as pl
from lightning.pytorch.tuner.tuning import Tuner

from src import utils, model, dataset

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(conf: DictConfig) -> None:

    logger.info('JACK SKELETON! A starting point for future projects')
    root = Path(conf.paths.root)
    logger.info(f'Root = {root}')
    logger.info(f'> PIPELINE: \n{OmegaConf.to_yaml(conf.pipeline)}')
    if conf.pipeline.show_full_conf:
        logger.info(f'> CONFS: \n{OmegaConf.to_yaml(conf)}')
    # logger.info(f'Optuna Version: {optuna.__version__}')

    # Config
    utils.set_seed(conf.env.seed)
    logger.info(f'Seed set to {conf.env.seed}')
    # device = utils.get_device(conf.env.device)
    logger.info(f'Device set to {conf.env.device}')

    # ---------------------------
    # DataLoaders
    # ---------------------------
    train_dl, val_dl, test_dl = dataset.get_dataloaders(conf)

    # ---------------------------
    # Lightning Module init
    # ---------------------------
    module = model.JackModule(model.Jack_Model, conf)
    logger.info(f"Initial LR: {module.lr}")

    weights_saver = model.WeightsSaver(weights_history_path=conf.paths.weights_history)
    trainer = pl.Trainer(
        max_epochs=conf.model.epochs,
        callbacks=[weights_saver],  # type: ignore
        # profiler="simple", # this line to check time durations and find bottlenecks
        # logger=False,  # we can use Hydra/Lightning logging later
        enable_checkpointing=False,
        # enable_progress_bar=False,
        # fast_dev_run = conf.pipeline.fast_debug_batch # run some batch of train, val and test searching bugs
    )
    try:
        tuner = Tuner(trainer) # type: ignore
        tuner.lr_find(module, train_dataloaders = train_dl) # type: ignore
        # now the lr is updated
    except:
        pass

    conf.optim[module.optim_name].lr = module.lr # update in memory
    logger.info(f'LR found: {module.lr}')
    OmegaConf.save(config=conf, 
                   f=Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)/'.hydra/config.yaml') # type: ignore # update in disk
    
    # ---------------------------
    # Optuna hyperparameter tuning
    # ---------------------------
    # breakpoint()
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, conf, train_dl, val_dl), n_trials=3)

    print(f"Best LR found: {study.best_params['lr']}")
    # if conf.pipeline.use_optuna:
    #     study = optuna.create_study(direction="minimize", study_name="lr_search")
    #     study.optimize(utils.objective,
    #                    n_trials=conf.optuna.trials)
    #     logger.info(f"Best hyperparameters: {study.best_params}")
    #     # Update conf with best values
    #     for k, v in study.best_params.items():
    #         conf.optim[conf.model.optim_name][k] = v

    # ---------------------------
    # Lightning training
    # ---------------------------
    # trainer.fit(module, train_dl)
    trainer.fit(module, train_dl, val_dl) # if validation step is needed

    # Access after training (last callback in list)
    all_weights = trainer.callbacks[0].epoch_weights # Dict: {0: tensor(shape), 1: tensor(shape), ...}
    glob_weights = torch.stack([w for w in all_weights.values()])
    glob_weights_path = conf.paths.glob_weights
    logger.info(f'Saving "glob_weights" in {glob_weights_path}')
    joblib.dump(glob_weights, glob_weights_path)

    # ---------------------------
    # Evaluation
    # ---------------------------
    trainer.test(module, test_dl)

def objective(trial, conf, train_dl, val_dl):
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    logger.info(f"Proposed LR by Optuna: {lr}")
    conf.optim[conf.model.optim_name].lr = lr
    module = model.JackModule(model.Jack_Model, conf)
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        # callbacks=[
        #     # This callback tells Optuna to kill the trial if it's performing poorly
        #     # PyTorchLightningPruningCallback(trial, monitor="val_loss")
        # ]
        # ... other trainer settings
    )
    
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    
    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    main()
    