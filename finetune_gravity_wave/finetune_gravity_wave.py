"""
Finetuning script for Momentum Flux downstream task.
Run using:

    torchrun finetune_gravity_wave.py --split uvtheta122
"""

import os
from typing import Literal
import argparse
import torch
import tqdm
import wandb
from datamodule import ERA5DataModule
from gravity_wave_model import UNetWithTransformer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from distributed import print0

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
device = f"cuda:{local_rank}"
dtype = torch.float32


def count_forward_pass_parameters(model):
    """Count the total number of parameters in a model that are used in the forward pass
    and have `requires_grad=True`.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: The total number of parameters used in the forward pass.
    """
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
    return total_params


def setup():
    """Initialize the process group for distributed training and set the CUDA device."""
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)


def cleanup():
    """Destroy the process group to clean up resources after training."""
    dist.destroy_process_group()


def train(cfg, rank):
    """Train the model using the specified configuration and rank.

    Args:
        cfg: The configuration object containing hyperparameters and paths.
        rank: The rank of the process in distributed training.
    """

    # Setup dataloaders
    vartype: Literal["uvtp122"] = cfg.vartype
    print0(f"Loading NetCDF data for variable type: {vartype}")

    datamodule_kwargs = dict(
        train_data_path=cfg.train_data_path,
        valid_data_path=cfg.valid_data_path,
        file_glob_pattern=cfg.file_glob_pattern,
    )

    # Setup Weights and Biases (wandb) logger
    if rank == 0:
        wandb.init(
            entity="define-entity",
            project="gravity-wave-flux",
            dir="logs",
            name=f"gwf_14_pre_{vartype}",
            mode=cfg.wandb_mode,
        )

    setup()

    # Initialize the data module and setup the dataset for training
    datamodule = ERA5DataModule(
        batch_size=cfg.batch_size,
        num_data_workers=cfg.num_data_workers,
        **datamodule_kwargs,
    )
    datamodule.setup(stage="fit")

    # Initialize the model and optimizer
    model: torch.nn.Module = UNetWithTransformer(
        lr=cfg.lr,
        hidden_channels=cfg.hidden_channels,
        in_channels={"uvtp122": 488}[vartype],
        out_channels={"uvtp122": 366}[vartype],
        n_lats_px=cfg.n_lats_px,
        n_lons_px=cfg.n_lons_px,
        in_channels_static=cfg.in_channels_static,
        mask_unit_size_px=cfg.mask_unit_size_px,
        patch_size_px=cfg.patch_size_px,
        device=device,
        ckpt_singular=cfg.singular_sharded_checkpoint,
    )
    optimizer: torch.optim.Optimizer = model.configure_optimizers()

    # Wrap model in DistributedDataParallel for multi-GPU training
    model = DDP(model.to(rank, dtype=dtype), device_ids=[rank])

    # Count and log the number of trainable parameters
    total_params = count_forward_pass_parameters(model)
    print0(f"TOTAL TRAINING PARAMETERS: {total_params:,}")

    # Start finetuning the model
    if rank == 0:
        print("Starting to finetune")

    for epoch in tqdm.trange(cfg.max_epochs):
        model.train()

        # Training loop
        pbar_train = tqdm.tqdm(
            iterable=datamodule.train_dataloader(), disable=(rank != 0)
        )
        for batch in pbar_train:
            # Move batch data to the appropriate device
            batch = {key: val.to(rank, dtype=dtype) for key, val in batch.items()}
            optimizer.zero_grad()

            # Forward pass
            y_hat: torch.Tensor = model.forward(batch)

            # Compute loss and metrics
            loss: torch.Tensor = torch.nn.functional.mse_loss(
                input=y_hat, target=batch["target"]
            )

            # Log training loss to wandb
            if rank == 0:
                pbar_train.set_postfix(ordered_dict={"train/loss": float(loss)})
                wandb.log(data={"train/loss": float(loss)})

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

        # Validation loop
        pbar_val = tqdm.tqdm(iterable=datamodule.val_dataloader(), disable=(rank != 0))
        with torch.no_grad():
            model.eval()

            for batch in pbar_val:
                # Move batch data to the appropriate device
                batch = {key: val.to(rank, dtype=dtype) for key, val in batch.items()}

                # Forward pass
                y_hat: torch.Tensor = model.forward(batch)

                # Compute validation loss and metrics
                val_loss: torch.Tensor = torch.nn.functional.mse_loss(
                    input=y_hat, target=batch["target"]
                )
                # Log validation loss to wandb
                if rank == 0:
                    pbar_val.set_postfix(ordered_dict={"val/loss": float(val_loss)})
                    wandb.log(data={"val/loss": float(val_loss)})

        # Save model checkpoint after each epoch
        if rank == 0:
            ckpt_path: str = (
                f"checkpoints/{vartype}/magnet-flux-{vartype}-epoch-{epoch:02d}-loss-{val_loss:.4f}.pt"
            )
            os.makedirs(name=os.path.dirname(p=ckpt_path), exist_ok=True)
            torch.save(obj=model.state_dict(), f=ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="uvtheta122",
        help="determines which dataset to use for training",
    )
    args = parser.parse_args()

    if args.split == "uvtp122":
        from config import get_cfg
    else:
        raise NotImplementedError

    cfg = get_cfg()
    train(cfg, local_rank)
