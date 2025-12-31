from omegaconf import OmegaConf, DictConfig
import hydra
import joblib
from pathlib import Path

import optuna
import torch
import pytorch_lightning as pl

from src import utils, model, dataset



logger = utils.get_logger("main")

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

    # DataSets
    train_ds = dataset.Jack_Dataset(
        n_samples=conf.data.train_samples, 
        shape=conf.data.array_dim)
    train_dl = train_ds.get_dataloader(conf.data.batch_size, conf.data.shuffle, conf.data.num_workers)
    
    val_ds = dataset.Jack_Dataset(
        n_samples=conf.data.val_samples, 
        shape=conf.data.array_dim)
    val_dl = val_ds.get_dataloader(conf.data.batch_size, conf.data.shuffle, conf.data.num_workers)

    test_ds = dataset.Jack_Dataset(
        n_samples=conf.data.test_samples, 
        shape=conf.data.array_dim)
    test_dl = test_ds.get_dataloader(conf.data.batch_size, conf.data.shuffle, conf.data.num_workers)

    # ---------------------------
    # Lightning Module init
    # ---------------------------
    module = model.JackModule(model.Jack_Model, conf)

    weights_saver = model.WeightsSaver()
    trainer = pl.Trainer(
        max_epochs=conf.model.epochs,
        callbacks=[weights_saver],
        # profiler="simple", # this line to check time durations and find bottlenecks
        # logger=False,  # we can use Hydra/Lightning logging later
        enable_checkpointing=False,
        # enable_progress_bar=False,
        # fast_dev_run = conf.pipeline.fast_debug_batch # run some batch of train, val and test searching bugs
    )

    # ---------------------------
    # Optuna hyperparameter tuning
    # ---------------------------
    if conf.pipeline.use_optuna:
        study = optuna.create_study(direction="minimize", study_name="lr_search")
        study.optimize(lambda trial: utils.objective(trial, conf, module, trainer),
                       n_trials=conf.optuna.trials)
        logger.info(f"Best hyperparameters: {study.best_params}")
        # Update conf with best values
        for k, v in study.best_params.items():
            conf.optim[conf.model.optim_name][k] = v

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


if __name__ == "__main__":
    main()
    