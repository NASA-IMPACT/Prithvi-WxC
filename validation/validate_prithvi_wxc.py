import argparse
from packaging.version import parse as parse_version
from typing import Dict
from collections.abc import Callable
import json
from pathlib import Path
import random
from functools import partial
from tqdm import tqdm

import numpy as np
import xarray as xr
import pandas as pd
import torch
from torch.utils.data import DataLoader

from PrithviWxC.dataloaders.merra2_rollout import Merra2RolloutDataset
from PrithviWxC.model import PrithviWxC
from validation.loss import NormalizedMSELoss
from validation.config import ExperimentConfig, get_config
from validation.reproducibility import hash_tensor, validate_inputs, validate_rmse
from validation.get_assets import get_model_data, get_data

# xarray changed the naming in this classmethod in 2023.11.0 from to_array to to_dataarray
# See https://github.com/pydata/xarray/releases/tag/v2023.11.0.
if parse_version(xr.__version__) >= parse_version('2023.11.0'):
    XARRAY_TO_DATAARRAY = True
else:
    XARRAY_TO_DATAARRAY = False


def preproc(batch: list[Dict], padding: dict[tuple[int]], rollout: bool=False) -> dict:
    '''
    Args:
        batch: List of training samples. Each sample should be a dictionary with keys 'sur_static', 'sur_vals', 'sur_tars', 'ulv_vals', 'ulv_tars', 'lead_time'.
            The tensors have shape:
                sur_static: Numpy array of shape (3, lat, lon). For each pixel (lat, lon), the first dimension indexes sin(lat), cos(lon), sin(lon).
                sur_vals: Torch tensor of shape (parameter, time, lat, lon).
                sur_tars: Torch tensor of shape (parameter, time, lat, lon).
                ulv_vals: Torch tensor of shape (parameter, level, time, lat, lon).
                ulv_tars: Torch tensor of shape (parameter, level, time, lat, lon).
                sur_climate: Torch tensor of shape (parameter, lat, lon)
                ulv_climate: Torch tensor of shape (parameter, level, lat, lon)
                lead_time: Integer.
        padding: Dictionary with keys 'level', 'lat', 'lon. For each the value is a tuple of length two indicating padding at the start and end of the relevant dimension.
    Returns:
        Dictionary with keys 'x', 'y', 'lead_time' and 'static'. Optionally there's also 'climate'. All are torch tensors. Shapes are as follows:
            x:                              batch, time, parameter, lat, lon
            y (if no rollout):              batch, parameter, lat, lon 
            y (if rollout):                 batch, time, parameter, lat, lon
            static:                         batch, parameter, lat, lon
            lead_time:                      batch
            climate (Optional; no rollout): batch, parameter, lat, lon
            climate (Optional; rollout):    batch, time, parameter, lat, lon
        Here, for x and y, 'parameter' is [surface parameter, upper level parameter x level].
        Similarly for the static information we have
            [sin(lat), cos(lon), sin(lon), cos(doy), sin(doy), cos(hod), sin(hod), ...]
        Where `...` marks additional static information such as lake cover.
    '''

    data_keys = set(batch[0].keys())

    essential_keys = {
        'sur_static',
        'sur_vals',
        'sur_tars',
        'ulv_vals',
        'ulv_tars',
        'input_time',
        'lead_time'
    }

    climate_keys = {
        'sur_climate',
        'ulv_climate',
    }

    all_keys = essential_keys | climate_keys

    if not essential_keys.issubset(data_keys):
        raise ValueError('Missing essential keys.')

    if not data_keys.issubset(all_keys):
        raise ValueError('Unexpected keys in batch.')

    # Bring all tensors from the batch into a single tensor
    upl_x = torch.empty((len(batch), *batch[0]['ulv_vals'].shape))
    upl_y = torch.empty((len(batch), *batch[0]['ulv_tars'].shape))

    sur_x = torch.empty((len(batch), *batch[0]['sur_vals'].shape))
    sur_y = torch.empty((len(batch), *batch[0]['sur_tars'].shape))

    sur_sta = torch.empty((len(batch), *batch[0]['sur_static'].shape))

    if rollout:
        lead_time = torch.empty(
            (len(batch), *batch[0]["lead_time"].shape),
            dtype=torch.float32
        )
    else:
        lead_time = torch.empty((len(batch),), dtype=torch.float32)
    input_time = torch.empty((len(batch),), dtype=torch.float32)

    for i, rec in enumerate(batch):
        sur_x[i] = torch.Tensor(rec['sur_vals'])
        sur_y[i] = torch.Tensor(rec['sur_tars'])

        upl_x[i] = torch.Tensor(rec['ulv_vals'])
        upl_y[i] = torch.Tensor(rec['ulv_tars'])

        sur_sta[i] = torch.Tensor(rec['sur_static'])

        lead_time[i] = rec['lead_time']
        input_time[i] = rec['input_time']

    # Reshape (batch, parameter, level, time, lat, lon) -> (batch, time, parameter, level, lat, lon)
    upl_x = upl_x.permute((0, 3, 1, 2, 4, 5))
    upl_y = upl_y.permute((0, 3, 1, 2, 4, 5))
    # Reshape (batch, parameter, time, lat, lon) -> (batch, time, parameter, lat, lon)
    sur_x = sur_x.permute((0, 2, 1, 3, 4))
    sur_y = sur_y.permute((0, 2, 1, 3, 4))

    # Pad
    padding_2d = (*padding['lon'], *padding['lat'])
    padding_3d = (*padding['lon'], *padding['lat'], *padding['level'])
    sur_x = torch.nn.functional.pad(sur_x, padding_2d, mode='constant', value=0)
    upl_x = torch.nn.functional.pad(upl_x, padding_3d, mode='constant', value=0)
    sur_y = torch.nn.functional.pad(sur_y, padding_2d, mode='constant', value=0)
    upl_y = torch.nn.functional.pad(upl_y, padding_3d, mode='constant', value=0)
    sur_sta = torch.nn.functional.pad(sur_sta, padding_2d, mode='constant', value=0)

    if not rollout:
        # Remove time for targets
        upl_y = torch.squeeze(upl_y, 1)
        sur_y = torch.squeeze(sur_y, 1)
        y = torch.cat(
            [
                sur_y,
                upl_y.reshape(upl_y.shape[0], upl_y.shape[1] * upl_y.shape[2], *upl_y.shape[3:]),
            ],
            dim=1,
        )
    else:
        y = torch.cat(
            [
                sur_y,
                upl_y.reshape(*upl_y.shape[:2], upl_y.shape[2]*upl_y.shape[3], *upl_y.shape[4:])
            ],
            dim=2,
        )

    # We stack along the combined parameter x level dimension
    x = torch.cat(
        [
            sur_x,
            upl_x.reshape(*upl_x.shape[:2], upl_x.shape[2] * upl_x.shape[3], *upl_x.shape[4:]),
        ],
        dim=2,
    )

    static = sur_sta

    if rollout:
        if climate_keys.issubset(data_keys):
            sur_climate = torch.empty((len(batch), *batch[0]["sur_climate"].shape))
            ulv_climate = torch.empty((len(batch), *batch[0]["ulv_climate"].shape))

            for i, rec in enumerate(batch):
                sur_climate[i] = torch.Tensor(rec["sur_climate"])
                ulv_climate[i] = torch.Tensor(rec["ulv_climate"])

            sur_climate = torch.nn.functional.pad(sur_climate, padding_2d, mode="constant", value=0)
            ulv_climate = torch.nn.functional.pad(ulv_climate, padding_3d, mode="constant", value=0)

            climate = torch.cat(
                [
                    sur_climate,
                    ulv_climate.reshape(*ulv_climate.shape[:2], -1, *ulv_climate.shape[4:]),
                ],
                dim=2,
            )

        target_time = torch.sum(lead_time).reshape(-1)

        return_value = {
            "x": x,
            "y": y,
            "lead_time": lead_time,
            "input_time": input_time,
            "target_time": target_time,
            "static": static,
        }

    else:
        if 'sur_climate' in batch[0].keys():
            sur_climate = torch.empty((len(batch), *batch[0]['sur_climate'].shape))
            ulv_climate = torch.empty((len(batch), *batch[0]['ulv_climate'].shape))
            for i, rec in enumerate(batch):
                sur_climate[i] = torch.Tensor(rec['sur_climate'])
                ulv_climate[i] = torch.Tensor(rec['ulv_climate'])
            sur_climate = torch.nn.functional.pad(sur_climate, padding_2d, mode='constant', value=0)
            ulv_climate = torch.nn.functional.pad(ulv_climate, padding_3d, mode='constant', value=0)

            climate = torch.cat(
                [
                    sur_climate,
                    ulv_climate.reshape(
                        ulv_climate.shape[0],
                        ulv_climate.shape[1] * ulv_climate.shape[2],
                        *ulv_climate.shape[3:],
                    ),
                ],
                dim=1,
            )

        return_value = {
            "x": x,
            "y": y,
            "lead_time": lead_time,
            "input_time": input_time,
            "static": static,
        }

    if 'sur_climate' in batch[0].keys():
        return_value['climate'] = climate

    return return_value


def assemble_input_scalers(config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    _kwargs_open_dataset = {'chunks' : None, 'cache' : True, 'engine' : 'h5netcdf'}

    with (
        xr.open_dataset(
            config.model.input_scalers_surface_path,
            **_kwargs_open_dataset
        ) as musigma_surface,
        xr.open_dataset(
            config.model.input_scalers_vertical_path,
            **_kwargs_open_dataset
        ) as musigma_vertical
    ):
        musigma_surface.load()
        musigma_vertical.load()

        mu_surface = musigma_surface[config.data.surface_vars].sel(statistic='mu')
        sigma_surface = musigma_surface[config.data.surface_vars].sel(statistic='sigma')
        mu_vertical = musigma_vertical[config.data.vertical_vars].sel(statistic='mu', lev=config.data.levels)
        sigma_vertical = musigma_vertical[config.data.vertical_vars].sel(statistic='sigma', lev=config.data.levels)

        if XARRAY_TO_DATAARRAY:
            mu_surface = mu_surface.to_dataarray(dim='parameter')
            sigma_surface = sigma_surface.to_dataarray(dim='parameter')
            mu_vertical = mu_vertical.to_dataarray(dim='parameter')
            sigma_vertical = sigma_vertical.to_dataarray(dim='parameter')
        else:
            mu_surface = mu_surface.to_array(dim='parameter')
            sigma_surface = sigma_surface.to_array(dim='parameter')
            mu_vertical = mu_vertical.to_array(dim='parameter')
            sigma_vertical = sigma_vertical.to_array(dim='parameter')

        mu = torch.cat(
            (
                torch.from_numpy(mu_surface.values),
                torch.from_numpy(mu_vertical.transpose('parameter', 'lev').values.reshape(-1)),
            ), dim=0
        ).to(dtype=torch.float32)

        sigma = torch.cat(
            (
                torch.from_numpy(sigma_surface.values),
                torch.from_numpy(sigma_vertical.transpose('parameter', 'lev').values.reshape(-1)),
            ), dim=0
        ).to(dtype=torch.float32)
        sigma = torch.clamp(sigma, 1e-4, 1e4)

        return mu, sigma

def assemble_static_input_scalers(config: ExperimentConfig, n_unscaled_parameters: int=7) -> tuple[torch.Tensor, torch.Tensor]:
    _kwargs_open_dataset = {'chunks' : None, 'cache' : True, 'engine' : 'h5netcdf'}

    with xr.open_dataset(
            config.model.input_scalers_surface_path,
            **_kwargs_open_dataset
        ) as musigma_surface:
        musigma_surface.load()

        mu_surface = musigma_surface[config.data.static_surface_vars].sel(statistic='mu')
        sigma_surface = musigma_surface[config.data.static_surface_vars].sel(statistic='sigma')

        if XARRAY_TO_DATAARRAY:
            mu_surface = mu_surface.to_dataarray(dim='parameter')
            sigma_surface = sigma_surface.to_dataarray(dim='parameter')
        else:
            mu_surface = mu_surface.to_array(dim='parameter')
            sigma_surface = sigma_surface.to_array(dim='parameter')

        mu = torch.cat(
            (
                torch.zeros((n_unscaled_parameters,), dtype=torch.float32),
                torch.from_numpy(mu_surface.values),
            ), dim=0
        ).to(dtype=torch.float32)

        sigma = torch.cat(
            (
                torch.ones((n_unscaled_parameters,), dtype=torch.float32),
                torch.from_numpy(sigma_surface.values),
            ), dim=0
        ).to(dtype=torch.float32)
        sigma = torch.clamp(sigma, 1e-4, 1e4)

        return mu, sigma

def assemble_output_scalers(config: ExperimentConfig) -> torch.Tensor:
    _kwargs_open_dataset = {'chunks' : None, 'cache' : True, 'engine' : 'h5netcdf'}

    if config.model.residual == 'none':
        _, sigma = assemble_input_scalers(config)
        variances = sigma**2
        return variances

    with (
        xr.open_dataset(
            config.model.output_scalers_surface_path,
            **_kwargs_open_dataset
        ) as scaler_surface,
        xr.open_dataset(
            config.model.output_scalers_vertical_path,
            **_kwargs_open_dataset
        ) as scaler_vertical
    ):
        scaler_surface = scaler_surface[config.data.surface_vars]
        scaler_vertical = scaler_vertical[config.data.vertical_vars].sel(lev=config.data.levels)

        if XARRAY_TO_DATAARRAY:
            scaler_surface = scaler_surface.to_dataarray(dim='parameter')
            scaler_vertical = scaler_vertical.to_dataarray(dim='parameter')
        else:
            scaler_surface = scaler_surface.to_array(dim='parameter')
            scaler_vertical = scaler_vertical.to_array(dim='parameter')

        variances = torch.cat(
            (
                torch.from_numpy(scaler_surface.values),
                torch.from_numpy(scaler_vertical.transpose('parameter', 'lev').values.reshape(-1)),
            ),
            dim=0
        ).to(dtype=torch.float32)

    # Looking through the numbers, we have values as extreme as 1e-59 and 7e6.
    variances = torch.clamp(variances, 1e-7, 1e7)

    return variances

def get_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    '''
    Args:
        config: Experiment configuration. Contains configuration parameters for model.
    Returns:
        Tuple of data loaders: (training loader, validation loader).
    '''

    valid_dataset = Merra2RolloutDataset(
        time_range=config.data.time_range_valid,
        roll_longitudes=0,
        data_path_surface=config.data.data_path_surface,
        data_path_vertical=config.data.data_path_vertical,
        climatology_path_surface=config.data.climatology_path_surface,
        climatology_path_vertical=config.data.climatology_path_vertical,
        surface_vars=config.data.surface_vars,
        static_surface_vars=config.data.static_surface_vars,
        vertical_vars=config.data.vertical_vars,
        levels=config.data.levels,
        input_time=config.data.input_time,
        lead_time=config.data.lead_time,
        positional_encoding = "fourier",
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(
            preproc,
            padding=config.data.padding,
            rollout=config.model.rollout,
        ),
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=True,
    )

    print(f"--> Validation batches: {len(valid_loader):,.0f}")
    print(f"--> Batch shape (UPPER): {valid_dataset.upper_shape}")
    print(f"--> Batch shape (SURFACE): {valid_dataset.surface_shape}")
    print(f"--> Validation samples: {len(valid_dataset):,.0f}")

    return valid_loader


def get_model(config: ExperimentConfig) -> torch.nn.Module:
    '''
    Args:
        config: Experiment configuration. Contains configuration parameters for model.
    Returns:
        The configured model.
    '''

    print("Creating the model.")

    input_mu, input_sigma = assemble_input_scalers(config)
    static_input_mu, static_input_sigma = assemble_static_input_scalers(config)
    output_var = assemble_output_scalers(config)

    model = PrithviWxC(
        in_channels = len(config.data.surface_vars)
        + len(config.data.levels) * len(config.data.vertical_vars),
        input_size_time = 2,
        in_channels_static = config.model.num_static_channels+len(config.data.static_surface_vars),
        input_scalers_mu = input_mu,
        input_scalers_sigma = input_sigma,
        input_scalers_epsilon = 0.0,
        static_input_scalers_mu = static_input_mu,
        static_input_scalers_sigma = static_input_sigma,
        static_input_scalers_epsilon = 0.0,
        output_scalers = torch.sqrt(output_var),
        n_lats_px = config.data.input_size_lat + sum(config.data.padding['lat']),
        n_lons_px = config.data.input_size_lon + sum(config.data.padding['lon']),
        patch_size_px = config.model.token_size,
        mask_unit_size_px = config.mask_unit_size,
        mask_ratio_inputs = config.mask_ratio_inputs,
        mask_ratio_targets = 0.0,
        embed_dim = config.model.embed_dim,
        n_blocks_encoder = config.model.n_blocks_encoder,
        n_blocks_decoder = config.model.n_blocks_decoder,
        mlp_multiplier = config.model.mlp_multiplier,
        n_heads = config.model.n_heads,
        dropout = config.model.dropout_rate,
        drop_path = config.model.drop_path,
        parameter_dropout = config.model.parameter_dropout,
        residual = config.model.residual,
        masking_mode = config.model.masking_mode,
        encoder_shifting = config.model.encoder_shift,
        decoder_shifting = config.model.decoder_shift,
        positional_encoding = config.model.__dict__.get('positional_encoding', 'absolute'),
        checkpoint_encoder = [int(i) for i in config.model.checkpoint_encoder],
        checkpoint_decoder = [int(i) for i in config.model.checkpoint_decoder],
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--> Model has {total_params:,.0f} params.")
    return model

def make_forecast(
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
    loss_func: Callable,
    weights: torch.Tensor,
    variable_names: list[str],
    rollout: int | None = None,
) -> (dict, xr.Dataset, float):
    """
    Args:
        batch: Data inputs. Keys:
            x (batch, time, parameter, lat, lon)
            y (batch, time, parameter, lat, lon)
            climate (batch, time, parameter, lat, lon)
            lead_time (batch)
            static (batch, parameter, lat, lon)
        model: The model.
    Returns:
        
    """
    input_hashes = {k : hash_tensor(v) for k, v in batch.items()}
    rmse = []
    loss = None

    print("Starting validation run.")

    dev_batch = {k: v.to("cuda") for k, v in batch.items()}

    xlast = dev_batch["x"][:,1]
    dev_batch["lead_time"] = dev_batch["lead_time"][...,0]
    dev_batch["statics"] = dev_batch["static"]
    dev_batch["climates"] = dev_batch["climate"]
    dev_batch["ys"] = dev_batch["y"]

    for step in tqdm(range(rollout)):
        dev_batch["static"] = dev_batch["statics"][:,step]
        dev_batch["climate"] = dev_batch["climates"][:,step]
        dev_batch["y"] = dev_batch["ys"][:,step]

        prediction = model(dev_batch)
        xcurr = prediction
        if loss is None:
            loss = loss_func(prediction, dev_batch)
        else:
            loss += loss_func(prediction, dev_batch)
        square_error = (prediction-dev_batch["y"])**2
        assert square_error.shape == weights.shape
        rmse.append(
            torch.sqrt(
                (square_error * weights).sum(dim=(0, 2, 3)) / weights.sum(dim=(0, 2, 3))
            ).cpu().numpy()
        )

        dev_batch["x"] = torch.cat((xlast[:,None], xcurr[:,None]), dim=1)
        xlast = xcurr

    loss = loss / rollout
    print("Validation run complete.")

    # Evaluation
    rmse = np.stack(rmse, axis=0)

    rmse = xr.DataArray(
        data = rmse,
        dims=("step", "variable"),
        coords={
            "step" : np.arange(rollout),
            "variable" : variable_names
        },
    )
    rmse = rmse.to_dataset(dim="variable")

    return input_hashes, rmse, loss.item()

def validate_forecast(input_hashes: dict, rmse: xr.Dataset):

    validation_object = Path("data/validation/validation_rmse.nc")
    if not validation_object.exists():
        rmse.to_netcdf(validation_object, engine="h5netcdf")
    else:
        reference_rmse = xr.open_dataset(validation_object)
        with pd.option_context('display.max_rows', None, "display.max_columns", 9):
            validated_rmses = validate_rmse(rmse, reference_rmse)

    validation_object = Path("data/validation/validation_data.json")
    if not validation_object.exists():
        with validation_object.open("w") as fp:
            json.dump(input_hashes, fp)
    else:
        with validation_object.open("r") as fp:
            reference_hashes = json.load(fp)
        validated_inputs = validate_inputs(input_hashes, reference_hashes)

    print(f"Validation result: inputs {validated_inputs}; RMSEs {validated_rmses}.")

def validation_run(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    loss_func: Callable,
    lats:np.array,
    variable_names: list[str] = None,
    rollout: int | None = None,
):
    assert len(validation_loader) == 1, "Expecting a single validation sample."
    assert rollout == 20, "Expecting 20 rollout steps."

    lats = torch.from_numpy(lats.reshape(1, 1, lats.shape[0], 1)).to("cuda")
    lats = torch.pi * lats / 180.
    weights = torch.cos(lats).expand(1, len(variable_names), 360, 576)

    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(validation_loader):
            assert batch["x"].shape[0] == 1, "Expecting batch size 1."

            input_hashes, rmse, loss = make_forecast(
                batch=batch,
                model=model,
                loss_func=loss_func,
                weights=weights,
                variable_names=variable_names,
                rollout=rollout,
            )

            validate_forecast(input_hashes, rmse)
            print(f"Validation loss: {loss}")
            print(f"Reference loss: 0.17125831544399261")

def main(config: ExperimentConfig) -> None:
    '''
    Main process, run within each individual GPU process.
    Args:
        config: Experiment configuration.
    '''

    # Construct detailed variable names including pressure levels for upper-level variables
    vertical_vars = config.data.vertical_vars
    surface_vars = config.data.surface_vars
    levels = config.data.levels
    variable_names = surface_vars + [
        f'{var}_level_{level}' for var in vertical_vars for level in levels
    ]

    print("Downloading validation data")
    get_data()
    print("Downloading model data")
    get_model_data()
    
    # Get dataloaders
    val_dl = get_dataloaders(config)

    print(f"--> Validation dataset with length {len(val_dl.dataset)}.")
    print(f"--> Validation data loader with length {len(val_dl)}.")
    print(f"--> Data samples: {val_dl.dataset.samples}.")

    torch.autograd.set_detect_anomaly(False)
    torch.jit.enable_onednn_fusion(True)
    
    random.seed(42 + 0)
    torch.cuda.manual_seed(42 + 0)
    torch.manual_seed(42 + 0)
    np.random.seed(42 + 0)

    # Create model
    model = get_model(config)

    rollout_arg = int(config.data.lead_time / -config.data.input_time)
    print(f"--> Forecast (rollout) validation with {rollout_arg} steps.")

    state_dict = torch.load("data/weights/prithvi.wxc.rollout.2300m.v1.pt", weights_only=False)
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda")

    # Loss functions
    lats = np.linspace(-90, 90, 361)
    lats = lats[: config.data.padding['lat'][1]]

    feature_weights = assemble_output_scalers(config)
    feature_weights = 1 / feature_weights
    val_loss_func = NormalizedMSELoss(
        lats=lats, feature_weights=feature_weights
    )

    validation_run(
        model=model,
        validation_loader=val_dl,
        loss_func=val_loss_func,
        lats=lats,
        variable_names=variable_names,
        rollout=rollout_arg,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Prithvi WxC validation')
    parser.add_argument("-c", "--config_path", type=str, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    main(config=get_config(args.config_path))
