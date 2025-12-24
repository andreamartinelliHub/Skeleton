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

    logger.info('Hello World!')
    root = Path(conf.paths.root)
    logger.info(f'Root = {root}')
    logger.info(f'> CONFS: \n{OmegaConf.to_yaml(conf)}')
    # logger.info(f'Optuna Version: {optuna.__version__}')

    # Config
    utils.set_seed(conf.env.seed)
    logger.info(f'Seed set to {conf.env.seed}')
    device = utils.get_device(conf.env.device)
    logger.info(f'Device set to {conf.env.device}')

    # DataSets
    train_ds = dataset.Jack_Dataset(
        n_samples=conf.data.train_samples, 
        shape=conf.data.array_dim)
    train_dl = train_ds.get_dataloader(conf.data.batch_size, conf.data.shuffle, conf.data.num_workers)

    val_ds = dataset.Jack_Dataset(
        n_samples=conf.data.train_samples, 
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
    # trainer = pl.Trainer(max_epochs=conf.model.epochs, accelerator="auto", devices="auto")
    trainer = pl.Trainer(
        # max_steps=1,
        max_epochs=1,
        logger=False,  # we can use Hydra/Lightning logging later
        enable_checkpointing=False,
        enable_progress_bar=False
    )

    # ---------------------------
    # Optuna hyperparameter tuning
    # ---------------------------
    if conf.pipeline.use_optuna:
        study = optuna.create_study(direction="minimize", study_name="lr_search")
        study.optimize(lambda trial: utils.objective(trial, conf, module, trainer, val_dl),
                       n_trials=conf.optuna.trials)
        logger.info(f"Best hyperparameters: {study.best_params}")
        # Update conf with best values
        for k, v in study.best_params.items():
            conf.optim[conf.model.optim_name][k] = v

    # ---------------------------
    # Lightning training
    # ---------------------------
    trainer.fit(module, train_dl)
    # ---------------------------
    # Evaluation
    # ---------------------------
    module.eval()
    # glob_weights = [module.model.linear.weight.detach().cpu().clone()]
    with torch.no_grad():
        for x, y in test_dl:
            y_hat = module(x.to(device))

    # #### IF NO PL
    # # Model, Optimizer, Loss
    # net = model.Jack_Model(conf.data.array_dim, conf.model.use_bias).to(device)
    # starting_net_weights = net.linear.weight.detach().cpu().clone()
    # opt = model.get_optimizer(net, conf.model.optim_name, conf.optim[conf.model.optim_name].lr)
    # loss_fn = model.get_loss_fn(conf.model.loss_name)

    # # Init Eval
    # _ = model.evaluate(
    #         model = net,
    #         eval_loader = test_dl,
    #         loss_fn = loss_fn,
    #         )
    
    # # Train
    # weights_evolution = model.train(
    #     model = net,
    #     train_loader = train_dl,
    #     optimizer = opt,
    #     loss_fn = loss_fn,
    #     epochs = conf.model.epochs,
    #     device = device,
    #     use_tqdm = conf.pipeline.use_tqdm
    # )
    
    # # End Eval
    # _ = model.evaluate(
    #         model = net,
    #         eval_loader = test_dl,
    #         loss_fn = loss_fn,
    #         )
    # glob_weights = [starting_net_weights] + weights_evolution
    
    history_path = conf.paths.history_metrics
    logger.info(f'Saving "glob_weights" in {history_path}')
    # joblib.dump(glob_weights, history_path)

if __name__ == "__main__":
    main()
    