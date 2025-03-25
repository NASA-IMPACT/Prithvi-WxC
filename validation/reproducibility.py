import numpy as np
import xarray as xr
import pandas as pd
import torch
import hashlib

def hash_tensor(t : torch.Tensor) -> str:
    """
    Args:
        t: Torch tensor.
    Returns:
        Hash string.
    """
    t = t.detach().cpu().numpy()
    hash = hashlib.blake2b(t.tobytes()).hexdigest()
    return hash

def validate_rmse(rmse: xr.Dataset, reference_rmse: xr.Dataset, threshold: float=5e-3) -> bool:
    """
    Args:
        rmse: RMSEs as recorded on local system. Dataset indexed by variable names. Sole dimension is forecast step.
        reference_rmse: RMSEs as recorded on reference system. Dataset indexed by variable names. Sole dimension is forecast step.
        threshold: Threshold value to use for validation.
    Returns:
        Boolean value indicating whether RMSEs are below validation threshold.
    """
    rmse_validation = pd.DataFrame.from_dict(
            {
            "T2M local" : rmse["T2M"],
            "T2M ref." : reference_rmse["T2M"],
            "U10M local" : rmse["U10M"],
            "U10M ref." : reference_rmse["U10M"],
        },
        orient="columns"
    )
    for c in ["T2M", "U10M"]:
        rmse_validation[f"{c} dev."] = (rmse_validation[f"{c} local"] - rmse_validation[f"{c} ref."]) / rmse_validation[f"{c} ref."]

    with pd.option_context('display.max_rows', None):
        print(rmse_validation)

    below_threshold = (rmse_validation[[f"{c} dev." for c in ["T2M", "U10M"]]] < threshold).all(axis=None)

    return below_threshold

def validate_inputs(validation_hashes: dict, reference_hashes: dict) -> bool:
    """
    Both inputs are dictionaries. Keys are as follows:
    - x
    - y
    - lead_time
    - input_time
    - target_time
    - static
    - climate
    Values are corresponding hashes.

    Args:
        validation_hashes: Dictionary with structure defined above.
        reference_hashes: Dictionary with structure defined above.
    Returns:
        Boolean value indicating whether inputs match.
    """
    data_matches = True

    for k in ["x", "y", "lead_time", "input_time", "target_time", "static", "climate"]:
        if validation_hashes[k] != reference_hashes[k]:
            print(f"Mismatch in inputs at key \"{k}\".")
            data_matches = False

    return data_matches
