import os
from argparse import Namespace
from typing import Optional

import yaml

class DataConfig:
    def __init__(
        self,
        surface_vars,
        static_surface_vars,
        vertical_vars,
        levels,
        time_range_train,
        time_range_valid,
        **kwargs,
    ):
        self.__dict__.update(kwargs)

        self.surface_vars = surface_vars
        self.static_surface_vars = static_surface_vars
        self.vertical_vars = vertical_vars
        self.levels = levels
        self.time_range_train = time_range_train
        self.time_range_valid = time_range_valid

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_argparse(args: Namespace):
        return DataConfig(**args.__dict__)

    def __str__(self):
        return (
            f"Input: {[self.input_size_time, self.input_size_level, self.input_size_lat, self.input_size_lon]}, "
            f"Output: {[self.target_size_time, self.input_size_level, self.input_size_lat, self.input_size_lon]}, "
            f"Lead time: {self.target_lead_time}, "
            f"Upper level vars: {self.upper_level_vars}, "
            f"Surface vars: {self.surface_vars}, "
        )

    def __repr__(self):
        return (
            f"Input: {[self.input_size_time, self.input_size_level, self.input_size_lat, self.input_size_lon]}, "
            f"Output: {[self.target_size_time, self.input_size_level, self.input_size_lat, self.input_size_lon]}, "
            f"Lead time: {self.target_lead_time}, "
            f"Upper level vars: {self.upper_level_vars}, "
            f"Surface vars: {self.surface_vars}, "
        )


class ModelConfig:
    def __init__(
        self,
        num_static_channels: Optional[int] = None,
        embed_dim: Optional[int] = None,
        token_size: Optional[tuple[int, int]] = None,
        n_blocks_encoder: Optional[int] = None,
        n_blocks_decoder: Optional[int] = None,
        mlp_multiplier: Optional[int] = None,
        n_heads: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        residual: Optional[bool] = False,
        train_loss: Optional[str] = None,
        val_loss: Optional[str] = None,
        **kwargs,
    ):
        self.__dict__.update(kwargs)

        self.num_static_channels = num_static_channels
        self.embed_dim = embed_dim
        self.token_size = token_size
        self.n_blocks_encoder = n_blocks_encoder
        self.n_blocks_decoder = n_blocks_decoder
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.residual = residual
        self.train_loss = train_loss
        self.val_loss = val_loss

        self.__dict__.update(kwargs)

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_argparse(args: Namespace):
        return ModelConfig(**args.__dict__)

    @property
    def encoder_d_ff(self):
        return int(self.enc_embed_size * self.mlp_ratio)

    @property
    def decoder_d_ff(self):
        return int(self.dec_embed_size * self.mlp_ratio)

    def __str__(self):
        return (
            f"Input channels: {self.num_input_channels}, "
            f"Encoder (L, H, E): {[self.enc_num_layers, self.enc_num_heads, self.enc_embed_size]}, "
            f"Decoder (L, H, E): {[self.dec_num_layers, self.dec_num_heads, self.dec_embed_size]}"
        )

    def __repr__(self):
        return (
            f"Input channels: {self.num_input_channels}, "
            f"Encoder (L, H, E): {[self.enc_num_layers, self.enc_num_heads, self.enc_embed_size]}, "
            f"Decoder (L, H, E): {[self.dec_num_layers, self.dec_num_heads, self.dec_embed_size]}"
        )


class ExperimentConfig:
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        batch_size: int,
        dl_num_workers: int,
        dl_prefetch_size: int,
        mask_unit_size: Optional[tuple[int]] = None,
        mask_ratio_inputs: Optional[float] = None,
        mask_ratio_targets: Optional[float] = None,
        **kwargs,
    ):
        # additional experiment parameters used in downstream tasks
        self.__dict__.update(kwargs)

        self.data = data_config
        self.model = model_config
        self.batch_size = batch_size
        self.dl_num_workers = dl_num_workers
        self.dl_prefetch_size = dl_prefetch_size
        self.mask_unit_size = mask_unit_size
        self.mask_ratio_inputs = mask_ratio_inputs
        self.mask_ratio_targets = mask_ratio_targets

    @property
    def path_checkpoint(self) -> str:
        if self.path_experiment == '':
            return os.path.join(self.path_weights, 'train', 'checkpoint.pt')
        else:
            return os.path.join(
                os.path.dirname(self.path_experiment), 'weights', 'train', 'checkpoint.pt'
            )

    @property
    def path_weights(self) -> str:
        return os.path.join(self.path_experiment, self.make_suffix_path(), "weights")

    @property
    def path_wandb(self) -> str:
        return os.path.join(self.path_experiment, self.make_suffix_path())

    def to_dict(self):
        d = self.__dict__.copy()
        d["model"] = self.model.to_dict()
        d["data"] = self.data.to_dict()

        return d

    @staticmethod
    def from_argparse(args: Namespace):
        return ExperimentConfig(
            data_config=DataConfig.from_argparse(args),
            model_config=ModelConfig.from_argparse(args),
            **args.__dict__,
        )

    @staticmethod
    def from_dict(params: dict):
        return ExperimentConfig(
            data_config=DataConfig(**params['data']),
            model_config=ModelConfig(**params['model']),
            **params,
        )

    def make_folder_name(self) -> str:
        param_folder = f"v1"
        return param_folder

    def make_suffix_path(self) -> str:
        return os.path.join(self.make_folder_name(), self.job_id)

    def __str__(self):
        return (
            f"ID: {self.job_id}, "
            f"Epochs: {self.num_epochs}, "
            f"Truncate train: {self.limit_steps_train}, "
            f"Truncate valid: {self.limit_steps_valid}, "
            f"Batch size: {self.batch_size}, "
            f"LR: {self.learning_rate}, "
            f"DL workers: {self.dl_num_workers}"
        )

    def __repr__(self):
        return (
            f"ID: {self.job_id}, "
            f"Epochs: {self.num_epochs}, "
            f"Truncate train: {self.limit_steps_train}, "
            f"Truncate valid: {self.limit_steps_valid}, "
            f"Batch size: {self.batch_size}, "
            f"LR: {self.learning_rate}, "
            f"DL workers: {self.dl_num_workers}"
        )

def get_config(config_path: str) -> ExperimentConfig:
    cfg = yaml.safe_load(open(config_path, 'r'))
    return ExperimentConfig.from_dict(cfg)