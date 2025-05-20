import os
import wandb
import argparse
from datetime import datetime

import torch

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from typing import List, Tuple, Dict, Any, Union, Optional

from config import config
from model.module import LightningSiameseNet
from data.dataloader import TripletDataset, create_triplet_dataloader

class MERTDataModule(L.LightningDataModule):
    def __init__(self, config: Dict, dataset_root_dir: str):
        super().__init__()
        dataset_splits = os.listdir(dataset_root_dir)
        assert "train" in dataset_splits and "test" in dataset_splits and "validation" in dataset_splits
        datasets = [
            TripletDataset(tracks_dir = os.path.join(dataset_root_dir, "train")),
            TripletDataset(tracks_dir = os.path.join(dataset_root_dir, "validation")),
            TripletDataset(tracks_dir = os.path.join(dataset_root_dir, "test")),
        ]
        dataset = torch.utils.data.ConcatDataset(datasets)
        print(f"Loaded {len(dataset)} MERT embedding tracks for training")

        assert config["train_split"] > 0 and config["train_split"] <= 1
        train_size = int(len(dataset) * config["train_split"])
        val_size = len(dataset) - train_size

        self.train_batch_size = config["train_batch_size"]
        self.num_workers = config["dataloader_num_workers"]
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return create_triplet_dataloader(
            self.train_dataset, 
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return create_triplet_dataloader(
            self.val_dataset, 
            batch_size=len(self.val_dataset),
            num_workers=self.num_workers,
        )

def train(
    config: Dict, 
    dataset_root_dir: str, 
    result_dir_basename: str, 
    load_ckpt_path: Optional[str] = None,
    num_gpus: Optional[int] = None,
    validate_only: bool = False,
):
    # create data module
    if not validate_only:
        data_module = MERTDataModule(config, dataset_root_dir)
        print("-- data module created --")
    else:
        val_dataset = TripletDataset(tracks_dir = os.path.join(dataset_root_dir, "validation"))
        val_dataloader = create_triplet_dataloader(
            val_dataset, 
            batch_size=len(val_dataset),
            num_workers=config["dataloader_num_workers"],
        )
        print("-- validation dataloader created --")

    # create neural net
    lightning_module = LightningSiameseNet(config)
    print("-- lightning module created --")

    # load model checkpoint
    if load_ckpt_path is not None:
        checkpoint = torch.load(load_ckpt_path, map_location="cpu")
        lightning_module.load_state_dict(checkpoint["state_dict"])

    # config multi-gpu training
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if num_gpus > 0 else "cpu"
    strategy = "ddp_find_unused_parameters_true" if num_gpus > 1 else "auto"
    print(f"-- Will be training with {accelerator}, a total of {num_gpus} GPUs --")

    # ckpt saving
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    result_dir = result_dir_basename + "-" + current_time
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    print(f"Checkpoints will be saved to {result_dir}")
    latest_checkpoint = ModelCheckpoint(
        dirpath=result_dir,
        filename="latest_model",
        save_last=True,
        save_weights_only=True,
        verbose=True,
    )
    min_loss_checkpoint = ModelCheckpoint(
        monitor="total_loss",
        dirpath=result_dir,
        filename="min_loss-{epoch}", 
        save_top_k=3,
        mode="min",
        save_weights_only=True,
        verbose=True,
    )
    max_accuracy_checkpoint = ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=result_dir,
        filename="max_accuracy-{epoch}",
        save_top_k=1,
        mode="max",
        save_weights_only=True,
        verbose=True,
    )

    # create logger
    wandb_logger = WandbLogger(project='melodysim_simaese_training')
    print("-- wandb logger created --")

    # Train using the lightning Trainer
    trainer = L.Trainer(
        max_epochs=config["max_epochs"], 
        check_val_every_n_epoch=config["check_val_every_n_epoch"], # every such training epochs, run validation
        callbacks=[latest_checkpoint, min_loss_checkpoint, max_accuracy_checkpoint],
        accelerator=accelerator,
        strategy=strategy,
        devices=num_gpus,
        logger=wandb_logger,
        log_every_n_steps=config["log_every_n_steps"]
    )

    if not validate_only:
        print("-- training started --")
        trainer.fit(lightning_module, datamodule=data_module)
        print("-- Training ended --")
    else:
        print("-- Validation started --")
        trainer.validate(lightning_module, dataloaders=val_dataloader)
        print("-- Validation ended --")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the feature extractor and classifier")
    parser.add_argument("-dataset-root-dir", type=str, help="Data folder with train/test/validation storing MERT encodings")
    parser.add_argument("-result-dir-basename", type=str, help="Data folder storing checkpoints")
    parser.add_argument("--load-ckpt-path", type=str, nargs="?", help="Path to load the checkpoint")
    parser.add_argument("--num-gpus", type=int, nargs="?", default=None, help="Specify the number of training gpus if needed")
    parser.add_argument("--validate-only", type=bool, default=False, help="If specified, then no training will be done")
    args = parser.parse_args()

    train(
        config, 
        dataset_root_dir=args.dataset_root_dir, 
        result_dir_basename=args.result_dir_basename,
        load_ckpt_path=args.load_ckpt_path,
        num_gpus=args.num_gpus,
        validate_only=args.validate_only,
    )
