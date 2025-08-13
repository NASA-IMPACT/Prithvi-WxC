"""
PrithviWxC.configs
==================

This module defines configurations for pre-trained instances of the Prithvi-WxC model.
"""
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import List, Optional

import torch
import yaml

from .model import PrithviWxC
from .dataloaders.merra2 import input_scalers, static_input_scalers, output_scalers


@dataclass
class PrithviWxCConfig:
    """
    Dataclass to hold configurations for pre-trained Prithvi-WxC models.
    """
    surface_vars: List[str]
    vertical_vars: List[str]
    static_surface_vars: List[str]
    levels: List[float]
    in_channels: int
    input_size_time: int
    in_channels_static: int
    input_scalers_epsilon: float
    static_input_scalers_epsilon: float
    n_lats_px: int
    n_lons_px: int
    patch_size_px: List[int]
    mask_unit_size_px: List[int]
    embed_dim: int
    n_blocks_encoder: int
    n_blocks_decoder: int
    mlp_multiplier: int
    n_heads: int
    dropout: float
    drop_path: float
    residual: str
    masking_mode: str
    encoder_shifting: bool
    decoder_shifting: bool
    parameter_dropout: float
    positional_encoding: str
    checkpoint_encoder: List[int] = None
    checkpoint_decoder: List[int] = None

    @property
    def n_vars_dynamic(self) -> int:
        return len(self.surface_vars) + len(self.vertical_vars) * len(self.levels)

    @property
    def n_vars_surface(self) -> int:
        return len(self.surface_vars)

    @property
    def n_vars_static(self) -> int:
        return len(self.static_vars)

    @property
    def n_vars_static_all(self) -> int:
        return len(self.all_static_vars)

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def all_static_vars(self) -> List[str]:
        static_vars = []
        if self.positional_encoding == "absolute":
            static_vars += ["SIN_LAT", "COS_LAT", "SIN_LAT"]
        else:
            static_vars += ["LAT", "LON"]

        static_vars += ["SIN_DOY", "COS_DOY", "SIN_HOD", "COS_HOD"]
        static_vars += self.static_surface_vars
        return static_vars

    @property
    def n_dynamic_vars(self) -> int:
        return len(self.surface_vars) + len(self.vertical_vars) * len(self.levels)


CONFIG_PATH = resources.files("PrithviWxC.config_files")

SMALL = PrithviWxCConfig(**yaml.safe_load(open(CONFIG_PATH.joinpath("small.yml"))))

LARGE = PrithviWxCConfig(**yaml.safe_load(open(CONFIG_PATH.joinpath("large.yml"))))

LARGE_ROLLOUT = PrithviWxCConfig(**yaml.safe_load(open(CONFIG_PATH.joinpath("large_rollout.yml"))))


def load_model(
        config: str,
        scaling_factor_dir: Path,
        weights: Optional[Path] = None
) -> PrithviWxC:
    """
    Load Prithvi-WxC model.

    Args:
        config: Name of the model configuration "large" or "small".
        scaling_factor_dir: The path containing the scaling factors.
        weights: Optional path pointing to the pre-trained weights to load.

    Retur
    """
    if not config.upper() in ["SMALL", "LARGE", "LARGE_ROLLOUT"]:
        raise ValueError(
            "'config' must be one of ['SMALL', 'LARGE', 'LARGE_ROLLOUT']"
        )
    args = asdict(globals().get(config.upper()))
    surface_vars = args.pop("surface_vars")
    static_surface_vars = args.pop("static_surface_vars")
    vertical_vars = args.pop("vertical_vars")
    levels = args.pop("levels")

    surf_in_scal_path = scaling_factor_dir / "climatology" / "musigma_surface.nc"
    vert_in_scal_path = scaling_factor_dir / "climatology" / "musigma_vertical.nc"
    surf_out_scal_path = scaling_factor_dir / "climatology" / "anomaly_variance_surface.nc"
    vert_out_scal_path = scaling_factor_dir / "climatology" / "anomaly_variance_vertical.nc"

    in_mu, in_sig = input_scalers(
        surface_vars,
        vertical_vars,
        levels,
        surf_in_scal_path,
        vert_in_scal_path,
    )
    output_sig = output_scalers(
        surface_vars,
        vertical_vars,
        levels,
        surf_out_scal_path,
        vert_out_scal_path,
    )
    static_mu, static_sig = static_input_scalers(
        surf_in_scal_path,
        static_surface_vars,
    )
    args["input_scalers_mu"] = in_mu
    args["input_scalers_sigma"] = in_sig
    args["static_input_scalers_mu"] = static_mu
    args["static_input_scalers_sigma"] = static_sig
    args["static_input_scalers_epsilon"] = 0.0
    args["output_scalers"] = output_sig ** 0.5
    args["mask_ratio_inputs"] = 0.0
    args["mask_ratio_targets"] = 0.0

    model = PrithviWxC(**args)

    if weights is not None:
        state_dict = torch.load(weights, weights_only=False)
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        model.load_state_dict(state_dict, strict=True)

    return model
