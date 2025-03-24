import functools as ft
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


def preproc(batch: list[dict], padding: dict[tuple[int]]) -> dict[str, Tensor]:
    """Prepressing function for MERRA2 Dataset

    Args:
        batch (dict): List of training samples, each sample should be a
            dictionary with the following keys::

            'sur_static': Numpy array of shape (3, lat, lon). For each pixel (lat, lon), the first dimension indexes sin(lat), cos(lon), sin(lon).
            'sur_vals': Torch tensor of shape (parameter, time, lat, lon).
            'sur_tars': Torch tensor of shape (parameter, time, lat, lon).
            'ulv_vals': Torch tensor of shape (parameter, level, time, lat, lon).
            'ulv_tars': Torch tensor of shape (parameter, level, time, lat, lon).
            'sur_climate': Torch tensor of shape (parameter, lat, lon)
            'ulv_climate': Torch tensor of shape (parameter, level, lat, lon)
            'lead_time': Integer.
            'input_time': Integer.

        padding: Dictionary with keys 'level', 'lat', 'lon', each of dim 2.

    Returns:
        Dictionary with the following keys::

            'x': [batch, time, parameter, lat, lon]
            'y': [batch, parameter, lat, lon]
            'static': [batch, parameter, lat, lon]
            'lead_time': [batch]
            'input_time': [batch]
            'climate (Optional)': [batch, parameter, lat, lon]

    Note:
        Here, for x and y, 'parameter' is [surface parameter, upper level,
        parameter x level]. Similarly for the static information we have
        [sin(lat), cos(lon), sin(lon), cos(doy), sin(doy), cos(hod), sin(hod),
        ...].
    """  # noqa: E501
    b0 = batch[0]
    nbatch = len(batch)
    data_keys = set(b0.keys())

    essential_keys = {
        "sur_static",
        "sur_vals",
        "sur_tars",
        "ulv_vals",
        "ulv_tars",
        "input_time",
        "lead_time",
    }

    climate_keys = {
        "sur_climate",
        "ulv_climate",
    }

    all_keys = essential_keys | climate_keys

    if not essential_keys.issubset(data_keys):
        raise ValueError("Missing essential keys.")

    if not data_keys.issubset(all_keys):
        raise ValueError("Unexpected keys in batch.")

    # Bring all tensors from the batch into a single tensor
    upl_x = torch.empty((nbatch, *b0["ulv_vals"].shape))
    upl_y = torch.empty((nbatch, *b0["ulv_tars"].shape))

    sur_x = torch.empty((nbatch, *b0["sur_vals"].shape))
    sur_y = torch.empty((nbatch, *b0["sur_tars"].shape))

    sur_sta = torch.empty((nbatch, *b0["sur_static"].shape))

    lead_time = torch.empty((nbatch,), dtype=torch.float32)
    input_time = torch.empty((nbatch,), dtype=torch.float32)

    for i, rec in enumerate(batch):
        sur_x[i] = rec["sur_vals"]
        sur_y[i] = rec["sur_tars"]

        upl_x[i] = rec["ulv_vals"]
        upl_y[i] = rec["ulv_tars"]

        sur_sta[i] = rec["sur_static"]

        lead_time[i] = rec["lead_time"]
        input_time[i] = rec["input_time"]

    return_value = {
        "lead_time": lead_time,
        "input_time": input_time,
    }

    # Reshape (batch, parameter, level, time, lat, lon) ->
    #  (batch, time, parameter, level, lat, lon)
    upl_x = upl_x.permute((0, 3, 1, 2, 4, 5))
    upl_y = upl_y.permute((0, 3, 1, 2, 4, 5))
    # Reshape (batch, parameter, time, lat, lon) ->
    #  (batch, time, parameter, lat, lon)
    sur_x = sur_x.permute((0, 2, 1, 3, 4))
    sur_y = sur_y.permute((0, 2, 1, 3, 4))

    # Pad
    padding_2d = (*padding["lon"], *padding["lat"])

    def pad2d(x):
        return torch.nn.functional.pad(x, padding_2d, mode="constant", value=0)

    padding_3d = (*padding["lon"], *padding["lat"], *padding["level"])

    def pad3d(x):
        return torch.nn.functional.pad(x, padding_3d, mode="constant", value=0)

    sur_x = pad2d(sur_x).contiguous()
    upl_x = pad3d(upl_x).contiguous()
    sur_y = pad2d(sur_y).contiguous()
    upl_y = pad3d(upl_y).contiguous()
    return_value["static"] = pad2d(sur_sta).contiguous()

    # Remove time for targets
    upl_y = torch.squeeze(upl_y, 1)
    sur_y = torch.squeeze(sur_y, 1)

    # We stack along the combined parameter x level dimension
    return_value["x"] = torch.cat(
        (sur_x, upl_x.view(*upl_x.shape[:2], -1, *upl_x.shape[4:])), dim=2
    )
    return_value["y"] = torch.cat(
        (sur_y, upl_y.view(upl_y.shape[0], -1, *upl_y.shape[3:])), dim=1
    )

    if climate_keys.issubset(data_keys):
        sur_climate = torch.empty((nbatch, *b0["sur_climate"].shape))
        ulv_climate = torch.empty((nbatch, *b0["ulv_climate"].shape))
        for i, rec in enumerate(batch):
            sur_climate[i] = rec["sur_climate"]
            ulv_climate[i] = rec["ulv_climate"]
        sur_climate = pad2d(sur_climate)
        ulv_climate = pad3d(ulv_climate)

        return_value["climate"] = torch.cat(
            (
                sur_climate,
                ulv_climate.view(nbatch, -1, *ulv_climate.shape[3:]),
            ),
            dim=1,
        )

    return return_value


def input_scalers(
    surf_vars: list[str],
    vert_vars: list[str],
    levels: list[float],
    surf_path: str | Path,
    vert_path: str | Path,
) -> tuple[Tensor, Tensor]:
    """Reads the input scalers

    Args:
        surf_vars: surface variables to be used.
        vert_vars: vertical variables to be used.
        levels: MERRA2 levels to use.
        surf_path: path to surface scalers file.
        vert_path: path to vertical level scalers file.

    Returns:
        mu (Tensor): mean values
        var (Tensor): varience values
    """
    with h5py.File(Path(surf_path), "r", libver="latest") as surf_file:
        stats = [x.decode().lower() for x in surf_file["statistic"][()]]
        mu_idx = stats.index("mu")
        sig_idx = stats.index("sigma")

        s_mu = torch.tensor([surf_file[k][()][mu_idx] for k in surf_vars])
        s_sig = torch.tensor([surf_file[k][()][sig_idx] for k in surf_vars])

    with h5py.File(Path(vert_path), "r", libver="latest") as vert_file:
        stats = [x.decode().lower() for x in vert_file["statistic"][()]]
        mu_idx = stats.index("mu")
        sig_idx = stats.index("sigma")

        lvl = vert_file["lev"][()]
        l_idx = [np.where(lvl == v)[0].item() for v in levels]

        v_mu = np.array([vert_file[k][()][mu_idx, l_idx] for k in vert_vars])
        v_sig = np.array([vert_file[k][()][sig_idx, l_idx] for k in vert_vars])

    v_mu = torch.from_numpy(v_mu).view(-1)
    v_sig = torch.from_numpy(v_sig).view(-1)

    mu = torch.cat((s_mu, v_mu), dim=0).to(torch.float32)
    sig = torch.cat((s_sig, v_sig), dim=0).to(torch.float32).clamp(1e-4, 1e4)
    return mu, sig


def static_input_scalers(
    scalar_path: str | Path, stat_vars: list[str], unscaled_params: int = 7
) -> tuple[Tensor, Tensor]:
    scalar_path = Path(scalar_path)

    with h5py.File(scalar_path, "r", libver="latest") as scaler_file:
        stats = [x.decode().lower() for x in scaler_file["statistic"][()]]
        mu_idx = stats.index("mu")
        sig_idx = stats.index("sigma")

        mu = torch.tensor([scaler_file[k][()][mu_idx] for k in stat_vars])
        sig = torch.tensor([scaler_file[k][()][sig_idx] for k in stat_vars])

    z = torch.zeros(unscaled_params, dtype=mu.dtype, device=mu.device)
    o = torch.ones(unscaled_params, dtype=sig.dtype, device=sig.device)
    mu = torch.cat((z, mu), dim=0).to(torch.float32)
    sig = torch.cat((o, sig), dim=0).to(torch.float32)

    return mu, sig.clamp(1e-4, 1e4)


def output_scalers(
    surf_vars: list[str],
    vert_vars: list[str],
    levels: list[float],
    surf_path: str | Path,
    vert_path: str | Path,
) -> Tensor:
    surf_path = Path(surf_path)
    vert_path = Path(vert_path)

    with h5py.File(surf_path, "r", libver="latest") as surf_file:
        svars = torch.tensor([surf_file[k][()] for k in surf_vars])

    with h5py.File(vert_path, "r", libver="latest") as vert_file:
        lvl = vert_file["lev"][()]
        l_idx = [np.where(lvl == v)[0].item() for v in levels]
        vvars = np.array([vert_file[k][()][l_idx] for k in vert_vars])
    vvars = torch.from_numpy(vvars).view(-1)

    var = torch.cat((svars, vvars), dim=0).to(torch.float32).clamp(1e-7, 1e7)

    return var


class SampleSpec:
    """
    A data class to collect the information used to define a sample.
    """

    def __init__(
        self,
        inputs: tuple[pd.Timestamp, pd.Timestamp],
        lead_time: int,
        target: pd.Timestamp | list[pd.Timestamp],
    ):
        """
        Args:
            inputs: Tuple of timestamps. In ascending order.
            lead_time: Lead time. In hours.
            target: Timestamp of the target. Can be before or after the inputs.
        """
        if not inputs[0] < inputs[1]:
            raise ValueError(
                "Timestamps in `inputs` should be in strictly ascending order."
            )

        self.inputs = inputs
        self.input_time = (inputs[1] - inputs[0]).total_seconds() / 3600
        self.lead_time = lead_time
        self.target = target

        self.times = [*inputs, target]
        self.stat_times = [inputs[-1]]

    @property
    def climatology_info(self) -> tuple[int, int]:
        """Get the required climatology info.

        :return: information required to obtain climatology data. Essentially
            this is the day of the year and hour of the day of the target
            timestamp, with the former restricted to the interval [1, 365].
        :rtype: tuple
        """
        return (min(self.target.dayofyear, 365), self.target.hour)

    @property
    def year(self) -> int:
        return self.inputs[1].year

    @property
    def dayofyear(self) -> int:
        return self.inputs[1].dayofyear

    @property
    def hourofday(self) -> int:
        return self.inputs[1].hour

    def _info_str(self) -> str:
        iso_8601 = "%Y-%m-%dT%H:%M:%S"

        return (
            f"Issue time: {self.inputs[1].strftime(iso_8601)}\n"
            f"Lead time: {self.lead_time} hours ahead\n"
            f"Input delta: {self.input_time} hours\n"
            f"Target time: {self.target.strftime(iso_8601)}"
        )

    @classmethod
    def get(cls, timestamp: pd.Timestamp, dt: int, lead_time: int):
        """Given a timestamp and lead time, generates a SampleSpec object
        describing the sample further.

        Args:
            timestamp: Timstamp of the sample, Ie this is the larger of the two
                input timstamps.
            dt: Time between input samples, in hours.
            lead_time: Lead time. In hours.

        Returns:
            SampleSpec
        """  # noqa: E501
        assert dt > 0, "dt should be possitive"
        lt = pd.to_timedelta(lead_time, unit="h")
        dt = pd.to_timedelta(dt, unit="h")

        if lead_time >= 0:
            timestamp_target = timestamp + lt
        else:
            timestamp_target = timestamp - dt + lt

        spec = cls(
            inputs=(timestamp - dt, timestamp),
            lead_time=lead_time,
            target=timestamp_target,
        )

        return spec

    def __repr__(self) -> str:
        return self._info_str()

    def __str__(self) -> str:
        return self._info_str()


class Merra2Dataset(Dataset):
    """MERRA2 dataset. The dataset unifies surface and vertical data as well as
    optional climatology.

    Samples come in the form of a dictionary. Not all keys support all
    variables, yet the general ordering of dimensions is
    parameter, level, time, lat, lon

    Note:
        Data is assumed to be in NetCDF files containing daily data at 3-hourly
        intervals. These follow the naming patterns
        MERRA2_sfc_YYYYMMHH.nc and MERRA_pres_YYYYMMHH.nc and can be located in
        two different locations. Optional climatology data comes from files
        climate_surface_doyDOY_hourHOD.nc and
        climate_vertical_doyDOY_hourHOD.nc.


    Note:
        `_get_valid_timestamps` assembles a set of all timestamps for which
        there is data (with hourly resolutions). The result is stored in
        `_valid_timestamps`. `_get_valid_climate_timestamps` does the same with
        climatology data and stores it in `_valid_climate_timestamps`.

        Based on this information, `samples` generates a list of valid samples,
        stored in `samples`. Here the format is::

            [
                [
                    (timestamp 1, lead time A),
                    (timestamp 1, lead time B),
                    (timestamp 1, lead time C),
                ],
                [
                    (timestamp 2, lead time D),
                    (timestamp 2, lead time E),
                ]
            ]

        That is, the outer list iterates over timestamps (init times), the
        inner over lead times. Only valid entries are stored.
    """

    valid_vertical_vars = [
        "CLOUD",
        "H",
        "OMEGA",
        "PL",
        "QI",
        "QL",
        "QV",
        "T",
        "U",
        "V",
    ]
    valid_surface_vars = [
        "EFLUX",
        "GWETROOT",
        "HFLUX",
        "LAI",
        "LWGAB",
        "LWGEM",
        "LWTUP",
        "PRECTOT",
        "PS",
        "QV2M",
        "SLP",
        "SWGNT",
        "SWTNT",
        "T2M",
        "TQI",
        "TQL",
        "TQV",
        "TS",
        "U10M",
        "V10M",
        "Z0M",
    ]
    valid_static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]

    valid_levels = [
        34.0,
        39.0,
        41.0,
        43.0,
        44.0,
        45.0,
        48.0,
        51.0,
        53.0,
        56.0,
        63.0,
        68.0,
        71.0,
        72.0,
    ]

    timedelta_input = pd.to_timedelta(3, unit="h")

    def __init__(
        self,
        time_range: tuple[str | pd.Timestamp, str | pd.Timestamp],
        lead_times: list[int],
        input_times: list[int],
        data_path_surface: str | Path,
        data_path_vertical: str | Path,
        climatology_path_surface: str | Path | None = None,
        climatology_path_vertical: str | Path | None = None,
        surface_vars: list[str] | None = None,
        static_surface_vars: list[str] | None = None,
        vertical_vars: list[str] | None = None,
        levels: list[float] | None = None,
        roll_longitudes: int = 0,
        positional_encoding: str = "absolute",
        rtype: type = np.float32,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Args:
            data_path_surface: Location of surface data.
            data_path_vertical: Location of vertical data.
            climatology_path_surface: Location of (optional) surface
                climatology.
            climatology_path_vertical: Location of (optional) vertical
                climatology.
            surface_vars: Surface variables.
            static_surface_vars: Static surface variables.
            vertical_vars: Vertical variables.
            levels: Levels.
            time_range: Used to subset data.
            lead_times: Lead times for generalized forecasting.
            roll_longitudes: Set to non-zero value to data by random amount
                along longitude dimension.
            position_encoding: possible values are
              ['absolute' (default), 'fourier'].
                'absolute' returns lat lon encoded in 3 dimensions using sine
                  and cosine
                'fourier' returns lat/lon to be encoded by model
                <any other key> returns lat/lon to be encoded by model
            rtype: numpy data type used during read
            dtype: torch data type of data output
        """

        self.time_range = (
            pd.to_datetime(time_range[0]),
            pd.to_datetime(time_range[1]),
        )
        self.lead_times = lead_times
        self.input_times = input_times
        self._roll_longitudes = list(range(roll_longitudes + 1))

        self._uvars = vertical_vars or self.valid_vertical_vars
        self._level = levels or self.valid_levels
        self._svars = surface_vars or self.valid_surface_vars
        self._sstat = static_surface_vars or self.valid_static_surface_vars
        self._nuvars = len(self._uvars)
        self._nlevel = len(self._level)
        self._nsvars = len(self._svars)
        self._nsstat = len(self._sstat)

        self.rtype = rtype
        self.dtype = dtype

        self.positional_encoding = positional_encoding

        self._data_path_surface = Path(data_path_surface)
        self._data_path_vertical = Path(data_path_vertical)

        self.dir_exists(self._data_path_surface)
        self.dir_exists(self._data_path_vertical)

        self._get_coordinates()

        self._climatology_path_surface = Path(climatology_path_surface) or None
        self._climatology_path_vertical = (
            Path(climatology_path_vertical) or None
        )
        self._require_clim = (
            self._climatology_path_surface is not None
            and self._climatology_path_vertical is not None
        )

        if self._require_clim:
            self.dir_exists(self._climatology_path_surface)
            self.dir_exists(self._climatology_path_vertical)
        elif (
            climatology_path_surface is None
            and climatology_path_vertical is None
        ):
            self._climatology_path_surface = None
            self._climatology_path_vertical = None
        else:
            raise ValueError(
                "Either both or neither of"
                "`climatology_path_surface` and"
                "`climatology_path_vertical` should be None."
            )

        if not set(self._svars).issubset(set(self.valid_surface_vars)):
            raise ValueError("Invalid surface variable.")

        if not set(self._sstat).issubset(set(self.valid_static_surface_vars)):
            raise ValueError("Invalid static surface variable.")

        if not set(self._uvars).issubset(set(self.valid_vertical_vars)):
            raise ValueError("Inalid vertical variable.")

        if not set(self._level).issubset(set(self.valid_levels)):
            raise ValueError("Invalid level.")

    @staticmethod
    def dir_exists(path: Path) -> None:
        if not path.is_dir():
            raise ValueError(f"Directory {path} does not exist.")

    @property
    def upper_shape(self) -> tuple:
        """Returns the vertical variables shape
        Returns:
            tuple: vertical variable shape in the following order::

                [VAR, LEV, TIME, LAT, LON]
        """
        return self._nuvars, self._nlevel, 2, 361, 576

    @property
    def surface_shape(self) -> tuple:
        """Returns the surface variables shape

        Returns:
            tuple: surafce shape in the following order::

                [VAR, LEV, TIME, LAT, LON]
        """
        return self._nsvars, 2, 361, 576

    def data_file_surface(self, timestamp: pd.Timestamp) -> Path:
        """Build the surfcae data file name based on timestamp

        Args:
            timestamp: a timestamp

        Returns:
            Path: constructed path
        """
        pattern = "MERRA2_sfc_%Y%m%d.nc"
        data_file = self._data_path_surface / timestamp.strftime(pattern)
        return data_file

    def data_file_vertical(self, timestamp: pd.Timestamp) -> Path:
        """Build the vertical data file name based on timestamp

        Args:
            timestamp: a timestamp

        Returns:
            Path: constructed path
        """
        pattern = "MERRA_pres_%Y%m%d.nc"
        data_file = self._data_path_vertical / timestamp.strftime(pattern)
        return data_file

    def data_file_surface_climate(
        self,
        timestamp: pd.Timestamp | None = None,
        dayofyear: int | None = None,
        hourofday: int | None = None,
    ) -> Path:
        """
        Returns the path to a climatology file based either on a timestamp or
        the dayofyear / hourofday combination.
        Args:
            timestamp: A timestamp.
            dayofyear: Day of the year. 1 to 366.
            hourofday: Hour of the day. 0 to 23.
        Returns:
            Path: Path to climatology file.
        """
        if timestamp is not None and (
            (dayofyear is not None) or (hourofday is not None)
        ):
            raise ValueError(
                "Provide either timestamp or both dayofyear and hourofday."
            )

        if timestamp is not None:
            dayofyear = min(timestamp.dayofyear, 365)
            hourofday = timestamp.hour

        file_name = f"climate_surface_doy{dayofyear:03}_hour{hourofday:02}.nc"
        data_file = self._climatology_path_surface / file_name
        return data_file

    def data_file_vertical_climate(
        self,
        timestamp: pd.Timestamp | None = None,
        dayofyear: int | None = None,
        hourofday: int | None = None,
    ) -> Path:
        """Returns the path to a climatology file based either on a timestamp
        or the dayofyear / hourofday combination.

        Args:
            timestamp: A timestamp. dayofyear: Day of the year. 1 to 366.
            hourofday: Hour of the day. 0 to 23.
        Returns:
            Path: Path to climatology file.
        """
        if timestamp is not None and (
            (dayofyear is not None) or (hourofday is not None)
        ):
            raise ValueError(
                "Provide either timestamp or both dayofyear and hourofday."
            )

        if timestamp is not None:
            dayofyear = min(timestamp.dayofyear, 365)
            hourofday = timestamp.hour

        file_name = f"climate_vertical_doy{dayofyear:03}_hour{hourofday:02}.nc"
        data_file = self._climatology_path_vertical / file_name
        return data_file

    def _get_coordinates(self) -> None:
        """
        Obtains the coordiantes (latitudes and longitudes) from a single data
        file.
        """
        timestamp = next(iter(self.valid_timestamps))

        file = self.data_file_surface(timestamp)
        with h5py.File(file, "r", libver="latest") as handle:
            self.lats = lats = handle["lat"][()].astype(self.rtype)
            self.lons = lons = handle["lon"][()].astype(self.rtype)

        deg_to_rad = np.pi / 180
        self._embed_lat = np.sin(lats * deg_to_rad).reshape(-1, 1)

        self._embed_lon = np.empty((2, 1, len(lons)), dtype=self.rtype)
        self._embed_lon[0, 0] = np.cos(lons * deg_to_rad)
        self._embed_lon[1, 0] = np.sin(lons * deg_to_rad)

    @ft.cached_property
    def lats(self) -> np.ndarray:
        timestamp = next(iter(self.valid_timestamps))

        file = self.data_file_surface(timestamp)
        with h5py.File(file, "r", libver="latest") as handle:
            return handle["lat"][()].astype(self.rtype)

    @ft.cached_property
    def lons(self) -> np.ndarray:
        timestamp = next(iter(self.valid_timestamps))

        file = self.data_file_surface(timestamp)
        with h5py.File(file, "r", libver="latest") as handle:
            return handle["lon"][()].astype(self.rtype)

    @ft.cached_property
    def position_signal(self) -> np.ndarray:
        """Generates the "position signal" that is part of the static
        features.

        Returns:
            Tensor: Torch tensor of dimension (parameter, lat, lon) containing
            sin(lat), cos(lon), sin(lon).
        """

        latitudes, longitudes = np.meshgrid(
            self.lats, self.lons, indexing="ij"
        )

        if self.positional_encoding == "absolute":
            latitudes = latitudes / 360 * 2.0 * np.pi
            longitudes = longitudes / 360 * 2.0 * np.pi
            sur_static = np.stack(
                [np.sin(latitudes), np.cos(longitudes), np.sin(longitudes)],
                axis=0,
            )
        else:
            sur_static = np.stack([latitudes / 360. * 2.0 * np.pi, longitudes / 360. * 2.0 * np.pi], axis=0)

        sur_static = sur_static.astype(self.rtype)

        return sur_static

    @ft.cached_property
    def valid_timestamps(self) -> set[pd.Timestamp]:
        """Generates list of valid timestamps based on available files. Only
        timestamps for which both surface and vertical information is available
        are considered valid.
        Returns:
            list: list of timestamps
        """

        s_glob = self._data_path_surface.glob("MERRA2_sfc_????????.nc")
        s_files = [os.path.basename(f) for f in s_glob]
        v_glob = self._data_path_surface.glob("MERRA_pres_????????.nc")
        v_files = [os.path.basename(f) for f in v_glob]

        s_re = re.compile(r"MERRA2_sfc_(\d{8}).nc\Z")
        v_re = re.compile(r"MERRA_pres_(\d{8}).nc\Z")
        fmt = "%Y%m%d"

        s_times = {
            (datetime.strptime(m[1], fmt))
            for f in s_files
            if (m := s_re.match(f))
        }
        v_times = {
            (datetime.strptime(m[1], fmt))
            for f in v_files
            if (m := v_re.match(f))
        }

        times = s_times.intersection(v_times)

        # Each file contains a day at 3 hour intervals
        times = {
            t + timedelta(hours=i) for i in range(0, 24, 3) for t in times
        }

        start_time, end_time = self.time_range
        times = {pd.Timestamp(t) for t in times if start_time <= t <= end_time}

        return times

    @ft.cached_property
    def valid_climate_timestamps(self) -> set[tuple[int, int]]:
        """Generates list of "timestamps" (dayofyear, hourofday) for which
        climatology data is present. Only instances for which surface and
        vertical data is available are considered valid.
        Returns:
            list: List of tuples describing valid climatology instances.
        """
        if not self._require_clim:
            return set()

        s_glob = self._climatology_path_surface.glob(
            "climate_surface_doy???_hour??.nc"
        )
        s_files = [os.path.basename(f) for f in s_glob]

        v_glob = self._climatology_path_vertical.glob(
            "climate_vertical_doy???_hour??.nc"
        )
        v_files = [os.path.basename(f) for f in v_glob]

        s_re = re.compile(r"climate_surface_doy(\d{3})_hour(\d{2}).nc\Z")
        v_re = re.compile(r"climate_vertical_doy(\d{3})_hour(\d{2}).nc\Z")

        s_times = {
            (int(m[1]), int(m[2])) for f in s_files if (m := s_re.match(f))
        }
        v_times = {
            (int(m[1]), int(m[2])) for f in v_files if (m := v_re.match(f))
        }

        times = s_times.intersection(v_times)

        return times

    def _data_available(self, spec: SampleSpec) -> bool:
        """
        Checks whether data is available for a given SampleSpec object. Does so
        using the internal sets with available data previously constructed. Not
        by checking the file system.
        Args:
            spec: SampleSpec object as returned by SampleSpec.get
        Returns:
            bool: if data is availability.
        """
        valid = set(spec.times).issubset(self.valid_timestamps)

        if self._require_clim:
            sci = spec.climatology_info
            ci = set(sci) if isinstance(sci, list) else set([sci])  # noqa: C405
            valid &= ci.issubset(self.valid_climate_timestamps)

        return valid

    @ft.cached_property
    def samples(self) -> list[tuple[pd.Timestamp, int, int]]:
        """
        Generates list of all valid samlpes.
        Returns:
            list: List of tuples (timestamp, input time, lead time).
        """
        valid_samples = []
        dts = [(it, lt) for it in self.input_times for lt in self.lead_times]

        for timestamp in sorted(self.valid_timestamps):
            timestamp_samples = []
            for it, lt in dts:
                spec = SampleSpec.get(timestamp, -it, lt)

                if self._data_available(spec):
                    timestamp_samples.append((timestamp, it, lt))

            if timestamp_samples:
                valid_samples.append(timestamp_samples)

        return valid_samples

    def _to_torch(
        self,
        data: dict[str, Tensor | list[Tensor]],
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, Tensor | list[Tensor]]:
        out = {}
        for k, v in data.items():
            if isinstance(v, list):
                out[k] = [torch.from_numpy(x).to(dtype) for x in v]
            else:
                out[k] = torch.from_numpy(v).to(dtype)

        return out

    def _lat_roll(
        self, data: dict[str, Tensor | list[Tensor]], n: int
    ) -> dict[str, Tensor | list[Tensor]]:
        out = {}
        for k, v in data.items():
            if isinstance(v, list):
                out[k] = [torch.roll(x, shifts=n, dims=-1) for x in v]
            else:
                out[k] = torch.roll(v, shifts=n, dims=-1)

        return out

    def _read_static_data(
        self, file: str | Path, doy: int, hod: int
    ) -> np.ndarray:
        with h5py.File(file, "r", libver="latest") as handle:
            lats_surf = handle["lat"]
            lons_surf = handle["lon"]

            nll = (len(lats_surf), len(lons_surf))

            npos = len(self.position_signal)
            ntime = 4

            nstat = npos + ntime + self._nsstat
            data = np.empty((nstat, *nll), dtype=self.rtype)

            for i, key in enumerate(self._sstat, start=npos + ntime):
                data[i] = handle[key][()].astype(dtype=self.rtype)

        # [possition signal], cos(doy), sin(doy), cos(hod), sin(hod)
        data[0:npos] = self.position_signal
        data[npos + 0] = np.cos(2 * np.pi * doy / 366)
        data[npos + 1] = np.sin(2 * np.pi * doy / 366)
        data[npos + 2] = np.cos(2 * np.pi * hod / 24)
        data[npos + 3] = np.sin(2 * np.pi * hod / 24)

        return data

    def _read_surface(
        self, tidx: int, nll: tuple[int, int], handle: h5py.File
    ) -> np.ndarray:
        data = np.empty((self._nsvars, *nll), dtype=self.rtype)

        for i, key in enumerate(self._svars):
            data[i] = handle[key][tidx][()].astype(dtype=self.rtype)

        return data

    def _read_levels(
        self, tidx: int, nll: tuple[int, int], handle: h5py.File
    ) -> np.ndarray:
        lvls = handle["lev"][()]
        lidx = self._level_idxs(lvls)

        data = np.empty((self._nuvars, self._nlevel, *nll), dtype=self.rtype)

        for i, key in enumerate(self._uvars):
            data[i] = handle[key][tidx, lidx][()].astype(dtype=self.rtype)

        return np.ascontiguousarray(np.flip(data, axis=1))

    def _level_idxs(self, lvls):
        lidx = [np.argwhere(lvls == int(lvl)).item() for lvl in self._level]
        return sorted(lidx)

    @staticmethod
    def _date_to_tidx(date: datetime | pd.Timestamp, handle: h5py.File) -> int:
        if isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()

        time = handle["time"]

        t0 = time.attrs["begin_time"][()].item()
        d0 = f"{time.attrs['begin_date'][()].item()}"

        offset = datetime.strptime(d0, "%Y%m%d")

        times = [offset + timedelta(minutes=int(t + t0)) for t in time[()]]
        return times.index(date)

    def _read_data(
        self, file_pair: tuple[str, str], date: datetime
    ) -> dict[str, np.ndarray]:
        s_file, v_file = file_pair

        with h5py.File(s_file, "r", libver="latest") as shandle:
            lats_surf = shandle["lat"]
            lons_surf = shandle["lon"]

            nll = (len(lats_surf), len(lons_surf))

            tidx = self._date_to_tidx(date, shandle)

            sdata = self._read_surface(tidx, nll, shandle)

        with h5py.File(v_file, "r", libver="latest") as vhandle:
            lats_vert = vhandle["lat"]
            lons_vert = vhandle["lon"]

            nll = (len(lats_vert), len(lons_vert))

            tidx = self._date_to_tidx(date, vhandle)

            vdata = self._read_levels(tidx, nll, vhandle)

        data = {"vert": vdata, "surf": sdata}

        return data

    def _read_climate(
        self, file_pair: tuple[str, str]
    ) -> dict[str, np.ndarray]:
        s_file, v_file = file_pair

        with h5py.File(s_file, "r", libver="latest") as shandle:
            lats_surf = shandle["lat"]
            lons_surf = shandle["lon"]

            nll = (len(lats_surf), len(lons_surf))

            sdata = np.empty((self._nsvars, *nll), dtype=self.rtype)

            for i, key in enumerate(self._svars):
                sdata[i] = shandle[key][()].astype(dtype=self.rtype)

        with h5py.File(v_file, "r", libver="latest") as vhandle:
            lats_vert = vhandle["lat"]
            lons_vert = vhandle["lon"]

            nll = (len(lats_vert), len(lons_vert))

            lvls = vhandle["lev"][()]
            lidx = self._level_idxs(lvls)

            vdata = np.empty(
                (self._nuvars, self._nlevel, *nll), dtype=self.rtype
            )

            for i, key in enumerate(self._uvars):
                vdata[i] = vhandle[key][lidx][()].astype(dtype=self.rtype)

        data = {
            "vert": np.ascontiguousarray(np.flip(vdata, axis=1)),
            "surf": sdata,
        }

        return data

    def get_data_from_sample_spec(
        self, spec: SampleSpec
    ) -> dict[str, Tensor | int | float]:
        """Loads and assembles sample data given a SampleSpec object.

        Args:
            spec (SampleSpec): Full details regarding the data to be loaded
        Returns:
            dict: Dictionary with the following keys::

                'sur_static': Torch tensor of shape [parameter, lat, lon]. For
                each pixel (lat, lon), the first 7 dimensions index sin(lat),
                cos(lon), sin(lon), cos(doy), sin(doy), cos(hod), sin(hod).
                Where doy is the day of the year [1, 366] and hod the hour of
                the day [0, 23].
                'sur_vals': Torch tensor of shape [parameter, time, lat, lon].
                'sur_tars': Torch tensor of shape [parameter, time, lat, lon].
                'ulv_vals': Torch tensor of shape [parameter, level, time, lat, lon].
                'ulv_tars': Torch tensor of shape [parameter, level, time, lat, lon].
                'sur_climate': Torch tensor of shape [parameter, lat, lon].
                'ulv_climate': Torch tensor of shape [paramter, level, lat, lon].
                'lead_time': Float.
                'input_time': Float.

        """  # noqa: E501

        # We assemble the unique timestamps for which we need data.
        vals_required = {*spec.times}
        stat_required = {*spec.stat_times}

        # We assemble the unique data files from which we need value data
        vals_file_map = defaultdict(list)
        for t in vals_required:
            data_files = (
                self.data_file_surface(t),
                self.data_file_vertical(t),
            )
            vals_file_map[data_files].append(t)

        # We assemble the unique data files from which we need static data
        stat_file_map = defaultdict(list)
        for t in stat_required:
            data_files = (
                self.data_file_surface(t),
                self.data_file_vertical(t),
            )
            stat_file_map[data_files].append(t)

        # Load the value data
        data = {}
        for data_files, times in vals_file_map.items():
            for time in times:
                data[time] = self._read_data(data_files, time)

        # Combine times
        sample_data = {}

        input_upl = np.stack([data[t]["vert"] for t in spec.inputs], axis=2)
        sample_data["ulv_vals"] = input_upl

        target_upl = data[spec.target]["vert"]
        sample_data["ulv_tars"] = target_upl[:, :, None]

        input_sur = np.stack([data[t]["surf"] for t in spec.inputs], axis=1)
        sample_data["sur_vals"] = input_sur

        target_sur = data[spec.target]["surf"]
        sample_data["sur_tars"] = target_sur[:, None]

        # Load the static data
        data_files, times = stat_file_map.popitem()
        time = times[0].dayofyear, times[0].hour
        sample_data["sur_static"] = self._read_static_data(
            data_files[0], *time
        )

        # If required load the surface data
        if self._require_clim:
            ci_year, ci_hour = spec.climatology_info

            surf_file = self.data_file_surface_climate(
                dayofyear=ci_year,
                hourofday=ci_hour,
            )

            vert_file = self.data_file_vertical_climate(
                dayofyear=ci_year,
                hourofday=ci_hour,
            )

            clim_data = self._read_climate((surf_file, vert_file))

            sample_data["sur_climate"] = clim_data["surf"]
            sample_data["ulv_climate"] = clim_data["vert"]

        # Move the data from numpy to torch
        sample_data = self._to_torch(sample_data, dtype=self.dtype)

        # Optionally roll
        if len(self._roll_longitudes) > 0:
            roll_by = random.choice(self._roll_longitudes)
            sample_data = self._lat_roll(sample_data, roll_by)

        # Now that we have rolled, we can add the static data
        sample_data["lead_time"] = spec.lead_time
        sample_data["input_time"] = spec.input_time

        return sample_data

    def get_data(
        self, timestamp: pd.Timestamp, input_time: int, lead_time: int
    ) -> dict[str, Tensor | int]:
        """
        Loads data based on timestamp and lead time.
        Args:
            timestamp: Timestamp.
            input_time: time between input samples.
            lead_time: lead time.
         Returns:
            Dictionary with keys 'sur_static', 'sur_vals', 'sur_tars',
                'ulv_vals', 'ulv_tars', 'sur_climate', 'ulv_climate',
                'lead_time'.
        """
        spec = SampleSpec.get(timestamp, -input_time, lead_time)
        sample_data = self.get_data_from_sample_spec(spec)
        return sample_data

    def __getitem__(self, idx: int) -> dict[str, Tensor | int]:
        """
        Loads data based on sample index and random choice of sample.
        Args:
            idx: Sample index.
         Returns:
            Dictionary with keys 'sur_static', 'sur_vals', 'sur_tars',
                'ulv_vals', 'ulv_tars', 'sur_climate', 'ulv_climate',
                'lead_time', 'input_time'.
        """
        sample_set = self.samples[idx]
        timestamp, input_time, lead_time, *nsteps = random.choice(sample_set)
        sample_data = self.get_data(timestamp, input_time, lead_time)
        return sample_data

    def __len__(self):
        return len(self.samples)
