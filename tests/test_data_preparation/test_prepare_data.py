"""Test module for the data preparation."""

from datetime import datetime

import xarray as xr

from data_preparation.prepare_data import get_surface, load_raw_merra


def test_load_3d_vars():
    """Test if 3D variables are being created with zero difference of the original ones created on NAS."""

    upl_data = load_raw_merra(
        date=datetime(day=30, month=3, year=2020),
        merra_folder="tests/raw_data",
        variable_type="pres",
        extract_time=True,
        extract_levels=True,
    )
    gt = xr.open_dataset("tests/raw_data/MERRA_pres_20200330.nc4")
    for var in gt.data_vars:
        assert (
            gt[var] - upl_data[var]
        ).sum().values == 0.0, f"Error in variable: {var}"


def test_load_2d_vars():
    """Test if surface variables are being created with zero difference of the original ones created on NAS."""
    sur_data = get_surface(
        datetime(day=30, month=3, year=2020), "tests/raw_data", "tests/raw_data"
    )
    gt = xr.open_dataset("tests/raw_data/MERRA2_sfc_20200330.nc4")

    for var in gt.data_vars:
        assert (
            gt[var] - sur_data[var]
        ).sum().values == 0.0, f"Error in variable: {var}"


if __name__ == "__main__":
    test_load_3d_vars()
    test_load_2d_vars()
