"""
Prithvi-Wxc.download
====================

This module provides functionality to download and prepare input data for the Prithvi-WxC model.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import cache
import getpass
import logging
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union

import requests
import requests_cache
from huggingface_hub import hf_hub_download, snapshot_download
import numpy as np
from tqdm import tqdm
import xarray as xr


from .definitions import (
    VALID_SURFACE_VARS,
    VALID_STATIC_SURFACE_VARS,
    VALID_VERTICAL_VARS,
    VALID_LEVELS,
    NAN_VALS
)


LOGGER = logging.getLogger(__name__)


def filename_to_date(path: Path) -> datetime:
    """
    Extract time from MERRA-2 filename.

    Args:
        path: A Path object pointing to a MERRA-2 file.

    Return:
        A datetime object representing the timestamp of the file.
    """
    fname = path.name
    parts = fname.split(".")
    date = datetime.strptime(parts[-2], "%Y%m%d")
    return date


def get_previous_file(path: Path) -> Path:
    """
    Get path MERRA-2 file from previous day.
    """
    fname = path.name
    parts = fname.split(".")
    date = datetime.strptime(parts[-2], "%Y%m%d")
    date = date - timedelta(days=1)
    parts[-2] = "%Y%m%d"
    new_fname = date.strftime("%Y/%m/%d/" + ".".join(parts))
    return path.parent.parent.parent.parent / new_fname


def find_file_url(
        base_url: str,
        product_name: str,
        time: np.datetime64,
):
    """
    Find URL of MERRA-2 file accounting for changes in production stream.

    Args:
        base_urls: The stem of the URL where the files are located.
        product_name: The product name as used in the file name.
        time: A numpy datetime64 object specifying the time for which to download the file.

    Return:
        A string containing the URL of the desired MERRA-2 file.
    """
    if time is None:
        url = f"{base_url}/1980/"
        fname = rf"MERRA2_\d\d\d\.{product_name}\.00000000.nc4"
    else:
        date = time.astype("datetime64[s]").item()
        url = date.strftime(
            f"{base_url}/%Y/%m/"
        )
        fname = date.strftime(rf"MERRA2_\d\d\d\.{product_name}\.%Y%m%d\.nc4")
    regexp = re.compile(rf'href="({fname})"')
    with requests_cache.CachedSession() as session:
        response = session.get(url)
        response.raise_for_status()
        matches = regexp.findall(response.text)
    if len(matches) == 0:
        raise ValueError(
            "Found no matching file in %s.",
            url
        )
    return url + matches[1]


MERRA2_PRODUCTS = {
    "M2I3NXASM": ("https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVASM.5.12.4", "inst3_3d_asm_Nv"),
    "M2I1NXASM": ("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I1NXASM.5.12.4", "inst1_2d_asm_Nx"),
    "M2T1NXLND": ("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXLND.5.12.4", "tavg1_2d_lnd_Nx"),
    "M2T1NXFLX": ("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4", "tavg1_2d_flx_Nx"),
    "M2T1NXRAD": ("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXRAD.5.12.4", "tavg1_2d_rad_Nx"),
    "CONST2DASM": ("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2C0NXASM.5.12.4",  "const_2d_asm_Nx"),
    "CONST2DCTM": ("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2C0NXCTM.5.12.4", "const_2d_ctm_Nx")
}


def get_merra_urls(time: np.datetime64) -> List[str]:
    """
    List MERRA2 URLS required to prepare the input data for a given time step.
    """
    with ThreadPoolExecutor(max_workers=5) as pool:
        tasks = []
        tasks.append(pool.submit(find_file_url, *MERRA2_PRODUCTS["M2I3NXASM"], time))
        tasks.append(pool.submit(find_file_url, *MERRA2_PRODUCTS["M2I1NXASM"], time))
        tasks.append(pool.submit(find_file_url, *MERRA2_PRODUCTS["M2T1NXLND"], time))
        tasks.append(pool.submit(find_file_url, *MERRA2_PRODUCTS["M2T1NXFLX"], time))
        tasks.append(pool.submit(find_file_url, *MERRA2_PRODUCTS["M2T1NXRAD"], time))

        m2i3nxasm_url = tasks[0].result()
        m2i1nxasm_url = tasks[1].result()
        m2t1nxlnd_url = tasks[2].result()
        m2t1nxflx_url = tasks[3].result()
        m2t1nxrad_url = tasks[4].result()

    return [
        m2i3nxasm_url,
        m2i1nxasm_url,
        m2t1nxlnd_url,
        m2t1nxflx_url,
        m2t1nxrad_url,
    ]


@cache
def get_credentials() -> Tuple[str, str]:
    """
    Retrieves user name and password for GES DISC server through from environment variables or
    through user interaction.
    """
    ges_disc_user = os.environ.get("GES_DISC_USER", None)
    ges_disc_pw = os.environ.get("GES_DISC_PASSWORD", None)

    if (not ges_disc_user is None) and (not ges_disc_pw is None):
        return ges_disc_user, ges_disc_pw

    username = input("GES DISC username: ")
    password = getpass.getpass("GES DISC password: ")
    return username, password


def download_merra_file(
        url: str,
        destination: Union[str, Path],
        force: bool = False,
        credentials: Optional[Tuple[str, str]] = None
) -> Path:
    """
    Download MERRA2 file if it not already exists.

    Args:
        url: String containing the URL of the file to download.
        destination: The folder to which to download the file.
        force: Set to 'True' to force download even if file exists locally.
        credentials: Credentials to authenticate to the NASA GES DISC service. Avoids query for username
            and password if provided.

    Return:
        A Path object pointing to the local file.
    """
    destination = Path(destination)
    destination.mkdir(exist_ok=True, parents=True)

    filename = url.split("/")[-1]
    destination = destination / filename

    if not force and destination.exists():

        try:
            data = xr.open_dataset(destination)
            data.close()
            return destination
        except Exception:
            destination.unlink()

    if credentials is None:
        auth = get_credentials()
    else:
        auth = credentials

    with requests.Session() as session:
        session.auth = auth
        redirect = session.get(url, auth=auth)
        response = session.get(redirect.url, auth=auth, stream=True)
        response.raise_for_status()

        with open(destination, "wb") as output:
            for chunk in response:
                output.write(chunk)

    return destination


def download_merra_files(
        time_steps: List[np.datetime64],
        destination: Union[str, Path] = Path(".")
) -> List[str]:
    """
    Download MERRA2 files required to prepare the input data for a sequence of time steps.

    Args:
         time_steps: List of timestamps for which MERRA-2 input data is required.
         destination: The path to which to download the files.

    Return:
         A lists containing the paths of the downloaded files.
    """
    urls = []
    for time in time_steps:
        urls += get_merra_urls(time)

    urls = list(set(urls))

    time = time.astype("datetime64[s]").item()
    files = []

    destination = Path(destination)

    auth = get_credentials()

    with ThreadPoolExecutor(max_workers=8) as pool:

        tasks = {}

        for url in urls:
            fname = url.split("/")[-1]
            file_date = filename_to_date(Path(fname))

            # Dynamic data
            year = file_date.year
            month = file_date.month
            day = file_date.day
            dest_dyn = Path(destination) / f"{year}/{month:02}/{day:02}"

            tasks[pool.submit(download_merra_file, url, dest_dyn, credentials=auth)] = url

        merra_const_url = find_file_url(*MERRA2_PRODUCTS["CONST2DCTM"], None)
        tasks[pool.submit(download_merra_file, merra_const_url, destination, credentials=auth)] = merra_const_url

        files = []
        for task in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading MERRA-2 files."):
            try:
                files.append(task.result())
            except Exception as exc:
                raise exc

    return files


def get_required_input_files(time: np.datetime64) -> List[str]:
    """
    Get required Prithvi-WxC input files for given time.

    Args:
        time: A numpy.datetime64 object defining the input time.

    Return:
        A list containing the required input file names..
    """
    date = time.astype("datetime64[s]").item()
    return [
        date.strftime("MERRA2_sfc_%Y%m%d.nc"),
        date.strftime("MERRA_pres_%Y%m%d.nc"),
    ]


def get_prithvi_wxc_climatology(
        time_steps: List[np.datetime64],
        climatology_dir: Path
):
    """
    Download climatology files for given times.

    Args:
        time_steps: A list of time steps for which to download the climatology
        climatology_dir: The path in which to store the climatology files.
    """
    for time in time_steps:
        date = time.astype("datetime64[s]").item()
        doy = (date - datetime(date.year, 1, 1)).days + 1
        hour = (date.hour // 3) * 3

        fname_vert = f"climatology/climate_vertical_doy{doy:03}_hour{hour:02}.nc"
        hf_hub_download(
            repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
            filename=fname_vert,
            local_dir=climatology_dir
        )
        fname_sfc = f"climatology/climate_surface_doy{doy:03}_hour{hour:02}.nc"
        hf_hub_download(
            repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
            filename=fname_sfc,
            local_dir=climatology_dir
        )


def get_prithvi_wxc_scaling_factors(
        scaling_factor_dir: Path
        ):
    """
    Download scaling factor for the Prithvi-WxC model.
    """

    scale_in_surf = "musigma_surface.nc"
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=f"climatology/{scale_in_surf}",
        local_dir=scaling_factor_dir,
    )
    scale_in_vert = "musigma_vertical.nc"
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=f"climatology/{scale_in_vert}",
        local_dir=scaling_factor_dir,
    )
    scale_out_surf = "anomaly_variance_surface.nc"
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=f"climatology/{scale_out_surf}",
        local_dir=scaling_factor_dir,
    )
    scale_out_vert = "anomaly_variance_vertical.nc"
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=f"climatology/{scale_out_vert}",
        local_dir=scaling_factor_dir,
    )


def extract_prithvi_wxc_input_data(
        time: np.datetime64,
        merra_data_path: Path,
        input_data_path: Path,
        force: bool = False
):
    """
    Download and prepare Prithvi-WxC input data for a single time step.

    Args:
        time: A datetime object specifying the day for which to extract the data.
        merra_data_path: A Path object pointing to the directory containing the raw MERRA-2 data.
        input_data_path: The directory to which to write the extracted input data.
        force: Set to 'True' to force input extract even the output files already exist.

    """
    merra_data_path = Path(merra_data_path)
    input_data_path = Path(input_data_path)

    # Nothing do to if files already exist.
    input_files = [input_data_path / input_file for input_file in get_required_input_files(time)]
    if not force and all([input_file.exists() for input_file in input_files]):
        return input_files
    date = time.astype("datetime64[s]").item()

    merra_files = sorted(list(merra_data_path.glob(date.strftime("**/MERRA2_*.%Y%m%d.nc4"))))
    const_files = sorted(list(merra_data_path.glob(date.strftime("**/MERRA2_101.const_2d_ctm_Nx.00000000.nc4"))))

    if len(merra_files) == 0 or len(const_files) == 0:
        raise ValueError(
            "Couldn't find the required MERRA-2 files for %s",
            time
        )

    merra_files = merra_files + const_files

    start_time = time.astype("datetime64[D]").astype("datetime64[h]")
    end_time = start_time + np.timedelta64(24, "h")
    time_steps = np.arange(start_time, end_time, np.timedelta64(3, "h"))

    all_data = []
    for path in merra_files:
        with xr.open_dataset(path) as data:
            if "const" in path.name:
                vars = [
                    var for var in VALID_STATIC_SURFACE_VARS if var in data.variables
                ]
            else:
                vars = [
                    var for var in VALID_VERTICAL_VARS + VALID_SURFACE_VARS if var in data.variables
                ]

            data = data[vars + ["time"]]
            if "lev" in data:
                data = data.loc[{"lev": np.flip(np.array(VALID_LEVELS))}]
            data = data.compute()

            for var in data:
                if var in NAN_VALS:
                    nan = NAN_VALS[var]
                    data[var].data[:] = np.nan_to_num(data[var].data, nan=nan)

            # For static data without time dependence simply collapse the time dimension.
            if data.time.size == 1:
                data = data[{"time": 0}]
            # For monthly static data, simply pick the right month
            elif data.time.size == 12:
                month = start_time.astype("datetime64[s]").item().month
                data = data[{"time": month - 1}]
            # Select time steps for three-hourly data, linearly interpolate for hourly data
            else:
                method = "nearest"

                if (data.time.data[0] - data.time.data[0].astype("datetime64[h]")) > 0:

                    prev_file = get_previous_file(path)
                    date = data.time.data[0].astype("datetime64[s]").item()
                    if prev_file.exists() and date.day == 1:
                        data = xr.concat([
                            xr.load_dataset(prev_file),
                            data
                        ], dim="time")

                    for var in data:
                        data[var].data[1:] = 0.5 * (data[var].data[1:] + data[var].data[:-1])
                    new_time = data.time.data - 0.5 * (data.time.data[1] -  data.time.data[0])
                    data = data.assign_coords(time=new_time)

                times = list(data.time.data)
                data = data.interp(time=time_steps, method=method)

            for var in data:
                if var in NAN_VALS:
                    data[var].data[:] = np.nan_to_num(data[var].data, nan=NAN_VALS[var], copy=True)

            all_data.append(data)


    data = xr.merge(all_data, compat="override")

    input_data_path.mkdir(exist_ok=True, parents=True)

    data_sfc = data[VALID_SURFACE_VARS + VALID_STATIC_SURFACE_VARS]
    encoding = {name: {"zlib": True} for name in data_sfc}
    data_sfc["time"] = (
        (data_sfc.time.astype("datetime64[m]").data - np.datetime64("2020-01-01", "m")).astype("timedelta64[m]").astype("int32")
    )
    encoding["time"] = {
        "dtype": "int32",
    }
    date = data.time.data[0].astype("datetime64[s]").item()
    output_file = date.strftime("MERRA2_sfc_%Y%m%d.nc")
    data_sfc.time.attrs = {
        "begin_time": 0,
        "begin_date": 20200101,
    }
    data_sfc.to_netcdf(input_data_path / output_file, encoding=encoding, engine="h5netcdf")


    data_pres = data[VALID_VERTICAL_VARS]
    encoding = {name: {"zlib": True} for name in data_pres}
    data_pres["time"] = (
        (data_pres.time.astype("datetime64[m]").data - np.datetime64("2020-01-01", "m")).astype("timedelta64[m]").astype("int32")
    )
    encoding["time"] = {
        "dtype": "int32",
    }
    data_pres.time.attrs = {
        "begin_time": 0,
        "begin_date": 20200101,
    }

    output_file = date.strftime(date.strftime("MERRA_pres_%Y%m%d.nc"))
    data_pres.to_netcdf(input_data_path / output_file, encoding=encoding, engine="h5netcdf")



def get_prithvi_wxc_input(
        time: np.datetime64,
        input_time_step: int,
        lead_time: int,
        input_data_dir: Path,
        download_dir: Optional[Path] = None,
):
    """
    Download and prepare Prithvi-WxC input data for a forecast initialized at a given time.

    To use this function you need a NASA EarthData account (https://urs.earthdata.nasa.gov/) and link
    the account to the NASA GES DISC data archive
    (https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ).

    Calling this function will trigger an interactive input querying username and password for accessing
    the NASA EarthData archive. The interactive input can be avoided by setting the 'GES_DISC_USER' and
    'GES_DISC_PASSWORD' environment variables.

    Args:
        time: The time at which the forecast is initialized.
        input_time: The time difference in hours to the previous input time step.
        lead_time: The lead time up to which the forecast is made.
        download_dir: The directory to use to store the raw MERRA 2 data.
        input_data_dir:
    """
    input_data_dir = Path(input_data_dir)
    if download_dir is None:
        tmpdir = TemporaryDirectory()
        download_dir = Path(tmpdir.name)
    else:
        tmpdir = None
        download_dir = Path(download_dir)

    try:
        input_times = [time - np.timedelta64(input_time_step, "h"), time]
        output_times = time + np.arange(
            input_time_step,
            lead_time + 1,
            input_time_step
        ).astype("timedelta64[h]")

        all_steps = list(input_times) + list(output_times)

        LOGGER.info("Downloading MERRA-2 files.")
        merra_files = download_merra_files(all_steps, download_dir / "raw")

        days = [time.astype("datetime64[s]").item() for time in all_steps]
        days = list(set([datetime(year=day.year, month=day.month, day=day.day) for day in days]))

        for day in tqdm(days, desc="Extracting input data"):
            extract_prithvi_wxc_input_data(
                np.datetime64(day.strftime("%Y-%m-%d")),
                download_dir,
                input_data_dir,
            )
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    LOGGER.info("Downloading climatology files.")
    get_prithvi_wxc_climatology(
        input_times + list(output_times),
        input_data_dir / "../climatology"
    )


REPO_IDS = {
    "large": "ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
    "large_rollout": "ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M-rollout"
}


def download_model_config(
        config_name: str,
        download_dir: Union[str, Path]
):
    """
    Download config.yml for a given pre-trained model from HuggingFace.

    Args:
        config_name: The name of the configuration ('large', 'large_rollout')
        download_dir: The directory to which to download the model config files.

    Return:
        The path of the downloaded file.
    """
    download_dir = Path(download_dir) / config_name

    repo_id = REPO_IDS.get(config_name, None)
    if repo_id is None:
        raise ValueError(
            f"Unknown config name '{config_name}'.",
        )

    return hf_hub_download(
        repo_id=repo_id,
        filename="config.yaml",
        local_dir=download_dir
    )


WEIGHT_FILE_NAMES = {
    "large": "prithvi.wxc.2300m.v1.pt",
    "large_rollout": "prithvi.wxc.rollout.2300m.v1.pt"
}


def download_model_weights(
        config_name: str,
        download_dir: Union[str, Path]
):
    """
    Download config.yml for a given pre-trained model from HuggingFace.

    Args:
        config_name: The name of the configuration ('large', 'large_rollout')
        download_dir: The directory to which to download the model config files.

    Return:
        The path of the downloaded file.
    """
    download_dir = Path(download_dir)

    if config_name.lower() == "small":
        weights = download_dir / "weights" / "small" / "prithvi.wxc.rollout.600m.v1.pt"
        if not weights.exists():
            raise ValueError(
                f"Expected the weights for the small model config at {weights} "
                "but the file doesn't exist."
            )
        return weights

    download_dir = Path(download_dir) / config_name

    repo_id = REPO_IDS.get(config_name, None)
    if repo_id is None:
        raise ValueError(
            f"Unknown config name '{repo_id}'.",
        )

    filename = WEIGHT_FILE_NAMES.get(config_name, None)
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=download_dir
    )
