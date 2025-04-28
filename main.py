import argparse
import logging
import os
from datetime import datetime

import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


from tagehr.models.crec_base import TagEHR
from data.dataset import build_dataloader
from utils import print_args, set_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument(
        "--dataset_name",
        default="mimic3",
        choices=["mimic3", "mimic4"],
        type=str,
    )
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_workers", default=4, type=int)
    parser.add_argument("--topk_smiles", default=1, type=int)
    # model
    # parser.add_argument("--model_name", default='CRecBase', type=str)
    # hyperparameter
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--w_pos", default=0.1, type=float)
    parser.add_argument("--w_neg", default=0.5, type=float)
    parser.add_argument("--w_reg", default=0.01, type=float)
    parser.add_argument(
        "--epsilon",
        default=0.2,
        type=float,
        help="epsilon controls the strength of perturbation",
    )
    # exp
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--interv_style",
        type=str,
        choices=["attention", "interpolation", "concat", "add"],
        default="concat",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--version", type=str)
    parser.add_argument("--dev", type=int, default=0)
    args = parser.parse_args()
    return args


def buil_model(args, dataset):
    return TagEHR(args, dataset, hidden_size=args.hidden_size)


def experiment(args):
    log_dir = f"run_logs/TagEHR/ver-{args.version}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    set_logger(log_dir)
    wandb_logger = WandbLogger(
        project="TagEHR",
        config=args,
        group=f"{args.model_name}",
        job_type=f"{args.version}",
        mode="online" if bool(args.wandb) else "disabled",
    )
    print_args(args)
    seed = L.seed_everything(args.seed)
    logger.info(f"Current PID: {os.getpid()}")
    logger.info(f"Global seed set to: {seed}")
    logger.info(f"CWD:{os.getcwd()}")

    dataset, train_loader, val_loader, test_loader = build_dataloader(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        dev=bool(args.dev),
        seed=seed,
    )
    logger.info("")
    logger.info(dataset.stat())

    model = buil_model(args, dataset)

    callbacks = []
    ckp_callback = ModelCheckpoint(
        dirpath=f"{log_dir}/checkpoint",
        monitor=f"val/ja",
        mode="max",
    )
    callbacks.append(ckp_callback)

    trainer = L.Trainer(
        default_root_dir=log_dir,
        callbacks=callbacks,
        devices=[args.device],
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        enable_progress_bar=bool(args.dev),
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)