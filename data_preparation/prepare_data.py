import argparse
import glob
import os
from datetime import datetime, timedelta

import xarray as xr
from data_preparation.aggregation import aggregate_merra


def get_default_merra_variables() -> dict:
    """Define MERRA-2 variables to be extracted.

    This function returns a dictionary containing the default MERRA-2 variables
    organized by category. Each category specifies the data collection and
    variables to extract.

    Categories:
        sfc: Surface variables from inst1_2d_asm_Nx collection
            - U10M, V10M: 10-meter wind components
            - T2M: 2-meter air temperature
            - QV2M: 2-meter specific humidity
            - PS: Surface pressure
            - TS: Surface skin temperature
            - TQI, TQL, TQV: Total ice, liquid water, and water vapor content

        pres: Pressure-level variables from inst3_3d_asm_Nv collection
            - U, V: Wind components
            - T: Temperature
            - QV: Specific humidity
            - OMEGA: Vertical velocity

        soil: Land surface variables from tavg1_2d_lnd_Nx collection
            - GWETROOT: Root zone soil wetness
            - LAI: Leaf area index

        flux: Surface flux variables from tavg1_2d_flx_Nx collection
            - EFLUX: Latent heat flux
            - HFLUX: Sensible heat flux
            - Z0M: Surface roughness

        pcp: Precipitation from tavg1_2d_flx_Nx collection
            - PRECTOT: Total precipitation

        radiation: Radiation variables from tavg1_2d_rad_Nx collection
            - LWGEM: Longwave ground emission
            - LWGAB: Longwave absorbed by ground
            - LWTUP: Upward longwave flux at top of atmosphere
            - SWGNT: Surface net downward shortwave flux
            - SWTNT: TOA net downward shortwave flux

        static1: Time-invariant fields from const_2d_asm_Nx collection
            - PHIS: Surface geopotential
            - FRLAND: Land fraction

        static2: Time-varying constants from const_2d_ctm_Nx collection
            - FROCEAN: Ocean fraction
            - FRACI: Ice fraction

        terrain_selected_levels: Selected vertical levels for terrain following

    Returns:
        dict: Dictionary containing MERRA-2 variable definitions by category
    """
    merra_variable_list = {
        "sfc": {
            "inst1_2d_asm_Nx": [
                "U10M",
                "V10M",
                "T2M",
                "QV2M",
                "SLP",
                "PS",
                "TS",
                "TQI",
                "TQL",
                "TQV",
            ]
        },
        "pres": {
            "inst3_3d_asm_Nv": [
                "U",
                "V",
                "T",
                "QV",
                "OMEGA",
                "PL",
                "H",
                "CLOUD",
                "QI",
                "QL",
            ]
        },
        "soil": {"tavg1_2d_lnd_Nx": ["GWETROOT", "LAI"]},
        "flux": {"tavg1_2d_flx_Nx": ["EFLUX", "HFLUX", "Z0M"]},
        "pcp": {"tavg1_2d_flx_Nx": ["PRECTOT"]},
        "radiation": {"tavg1_2d_rad_Nx": ["LWGEM", "LWGAB", "LWTUP", "SWGNT", "SWTNT"]},
        "static1": {"const_2d_asm_Nx": ["PHIS", "FRLAND"]},
        "static2": {"const_2d_ctm_Nx": ["FROCEAN", "FRACI"]},
        "terrain_selected_levels": [
            72,
            71,
            68,
            63,
            56,
            53,
            51,
            48,
            45,
            44,
            43,
            41,
            39,
            34,
        ],
    }
    return merra_variable_list


def load_static_merra(
    static_folder: str, variable_base_variable: str = "static1"
) -> xr.Dataset:
    """Extract static variables from MERRA-2 reanalysis data.

    This function loads static variables (like land fraction, terrain height etc.)
    from MERRA-2 reanalysis dataset. These variables typically don't change over time
    and are stored in special constant files.

    Args:
        variable_base_variable (str, optional): Type of static variable to extract.
            Can be either "static1" (for PHIS, FRLAND) or "static2" (for FROCEAN, FRACI).
            Defaults to "static1".

    Returns:
        xarray.Dataset: Dataset containing the requested static variables with
            dimensions [lat, lon]. Time dimension is dropped and any singleton
            dimensions are squeezed.

    Example:
        >>> static_data = extract_merra_static_vars("static1")
        >>> print(static_data.variables)
        {'PHIS': <xarray.Variable>, 'FRLAND': <xarray.Variable>}
    """
    variables = get_default_merra_variables()

    variable_kw = list(variables[variable_base_variable].keys())[0]
    merra_variables = list(variables[variable_base_variable].values())[0]
    static_data = xr.open_dataset(
        os.path.join(static_folder, f"MERRA2.{variable_kw}.00000000.nc4")
    )[merra_variables]
    return static_data.drop_vars("time").squeeze()


def load_raw_merra(
    date,
    merra_folder: str,
    variable_type: str,
    extract_time: bool = False,
    extract_levels: bool = False,
    aggregation: bool = False,
    aggregation_type: str = "sfc",
) -> xr.Dataset:
    """Load MERRA-2 variables for a specific day.

    This function loads variables from MERRA-2 data files for a given date. It searches for
    files matching the specified variable type and date pattern, reads them using xarray, and
    optionally filters by specific time steps and pressure levels.

    Args:
        date (datetime): Day to be processed.
        home_folder (str): Base directory containing MERRA-2 data files
        variable_type (str): Type of variables to extract (e.g., 'pres', 'sfc')
        extract_time (bool, optional): If True, extract data only for specific hours (0,3,6,...,21).
            Defaults to False.
        extract_levels (bool, optional): If True, extract data only for terrain-following pressure
            levels. Defaults to False.

    Returns:
        xarray.Dataset: Dataset containing the extracted MERRA-2 variables

    Raises:
        FileNotFoundError: If no files are found matching the date and variable type pattern

    Example:
        >>> data = extract_merra_vars(2020-03-03, '/path/to/merra2', 'pres',
        ...                          extract_time=True, extract_levels=True)
    """
    variables = get_default_merra_variables()
    variable_kw = list(variables[variable_type].keys())[0]
    merra_variables = list(variables[variable_type].values())[0]

    file_pattern = f"{merra_folder}/*/*{variable_kw}.{date.strftime('%Y%m%d')}*"
    print(file_pattern)

    files = sorted(glob.glob(file_pattern))
    # Add day before if exists
    preday = date - timedelta(days=1)
    db = glob.glob(f"{merra_folder}/*/*{variable_kw}.{preday.strftime('%Y%m%d')}*")
    has_preday = len(db) > 0
    files = sorted(files + db)

    print(files)
    if not files:
        raise FileNotFoundError(f"No files found for {date.strftime('%Y-%m-%d')}")

    data = xr.open_mfdataset(files)[merra_variables]
    if has_preday:
        data = data.sel(time=slice(f"{preday.strftime('%Y-%m-%d')} 21:00", None))

    if extract_time:
        data = data.sel(time=data["time"].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21]))

    if extract_levels:
        data = data.sel(lev=variables["terrain_selected_levels"])

    if aggregation:
        data = aggregate_merra(data, grouping_type=aggregation_type)

    if has_preday:
        data = data.sel(time=slice(f"{date.strftime('%Y-%m-%d')}", None))

    return data


def get_surface(
    date: datetime,
    merra_folder: str,
    static_folder: str,
) -> xr.Dataset:
    """Process and combine MERRA-2 surface-level data for a specific day.

    This function extracts and combines various surface-level variables from MERRA-2
    reanalysis data, including:
    - Surface variables (temperature, pressure, etc.)
    - Soil variables (moisture, temperature)
    - Surface fluxes (heat, moisture)
    - Precipitation variables
    - Radiation variables
    - Static surface properties

    The data is merged into a single dataset and saved as a NetCDF file.

    Args:
        date (datetime): Day to be processed.
        merra_folder (str): Path to folder containing time-dependent raw netCDF files.
        static_folder (str): Path to folder containing static raw netCDF files.

    Returns:
        xarray.Dataset with processed data.

    Raises:
        FileNotFoundError: If required MERRA-2 input files are not found

    Example:
        >>> process_sfc_data(2021-02-05, '/path/time_folder', 'path/static_folder')
    """

    # Process surface variables
    sfc_data = load_raw_merra(
        date=date, merra_folder=merra_folder, variable_type="sfc", extract_time=True
    )
    soil_data = load_raw_merra(
        date=date,
        merra_folder=merra_folder,
        variable_type="soil",
        aggregation=True,
        aggregation_type="sfc",
    )
    soil_data["GWETROOT"] = soil_data["GWETROOT"].fillna(1.0)
    soil_data["LAI"] = soil_data["LAI"].fillna(0.0)
    flux_data = load_raw_merra(
        date=date,
        merra_folder=merra_folder,
        variable_type="flux",
        aggregation=True,
        aggregation_type="sfc",
    )
    pcp_data = load_raw_merra(
        date=date,
        merra_folder=merra_folder,
        variable_type="pcp",
        aggregation=True,
        aggregation_type="pcp",
    )
    radiation_data = load_raw_merra(
        date=date,
        merra_folder=merra_folder,
        variable_type="radiation",
        aggregation=True,
        aggregation_type="sfc",
    )

    # Get static data
    static_data1 = load_static_merra(static_folder, "static1")
    static_data2 = (
        load_static_merra(static_folder, "static2")
        .isel(time=int(date.month) - 1)
        .squeeze()
    )

    # Merge all data
    return xr.merge(
        [
            sfc_data,
            soil_data,
            flux_data,
            pcp_data,
            radiation_data,
            static_data1,
            static_data2,
        ]
    )


def store_data(path: str, data: xr.Dataset):
    """Stores data in netCDF4 format."""

    if "lev" in data.coords:
        ds_rechunked = data.chunk({"lat": -1, "lon": -1, "lev": -1, "time": 1})
    else:
        ds_rechunked = data.chunk({"lat": -1, "lon": -1, "time": 1})
    encoding = {
        cvar: {
            "zlib": True,
            "complevel": 5,
            "chunksizes": [c[0] for c in ds_rechunked[cvar].chunks],
        }
        for cvar in data.data_vars
    }

    # Save to NetCDF4 with compression "chunksizes": [c[0] for c in ds_rechunked[cvar].chunks]
    ds_rechunked.to_netcdf(path, mode="w", encoding=encoding)


def process_day(date: datetime, output_folder: str, raw_merra: str, raw_static: str):
    """Process both surface and pressure level data for a specific day.

    This function processes MERRA-2 data for a given date by splitting the date string
    into components and calling the appropriate processing functions. Currently configured
    to process pressure level data only.

    Args:
        date (datetime): Date to be processed in daily resolution.
        output_folder (str): Path to folder where processed files will be saved.
        merra_folder (str): Path to folder containing time-dependent input raw files.
        static_folder (str): Path to folder containing static raw files.

    Returns:
        None

    Example:
        >>> process_day(2021-02-05, '/path/to/output', '/path/to/input_merra', '/path/to/input_static')
    """

    sur_data = get_surface(date, raw_merra, raw_static)
    store_data(f"{output_folder}/MERRA2_sfc_{date.strftime('%Y%m%d')}.nc", sur_data)
    print(f"Successfully processed surface data for {date.strftime('%Y-%m-%d')}")

    upl_data = load_raw_merra(
        date=date,
        merra_folder=raw_merra,
        variable_type="pres",
        extract_time=True,
        extract_levels=True,
    )
    store_data(f"{output_folder}/MERRA_pres_{date.strftime('%Y%m%d')}.nc", upl_data)
    print(f"Successfully processed pressure level data for {date.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        "-d",
        type=lambda s: datetime.strptime(s, "%Y%m%d"),
        help="Date in the format YYYYMMDD.",
    )
    parser.add_argument(
        "--input_folder",
        "-i",
        type=str,
        help="Input folder containing time-dependent raw netCDF files to be processed.",
    )
    parser.add_argument(
        "--static_folder",
        "-s",
        type=str,
        help="Input folder containing static raw netCDF files to be processed.",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        help="Path to the output folder.",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Process the data for specified date
    print(f"Processing date: {args.date.strftime('%Y-%m-%d')}")
    process_day(
        date=args.date,
        output_folder=args.output_folder,
        raw_merra=args.input_folder,
        raw_static=args.static_folder,
    )
