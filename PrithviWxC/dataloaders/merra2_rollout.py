import functools as ft
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from PrithviWxC.dataloaders.merra2 import Merra2Dataset, SampleSpec

def preproc(
    batch: list[dict[str, int | float | Tensor]], padding: dict[tuple[int]]
) -> dict[str, Tensor]:
    """Prepressing function for MERRA2 Dataset

    Args:
        batch (dict): List of training samples, each sample should be a
            dictionary with the following keys::

            'sur_static': Numpy array of shape (3, lat, lon). For each pixel (lat, lon), the first dimension indexes sin(lat), cos(lon), sin(lon).
            'sur_vals': Torch tensor of shape (parameter, time, lat, lon).
            'sur_tars': Torch tensor of shape (parameter, time, lat, lon).
            'ulv_vals': Torch tensor of shape (parameter, level, time, lat, lon).
            'ulv_tars': Torch tensor of shape (parameter, level, time, lat, lon).
            'sur_climate': Torch tensor of shape (nstep, parameter, lat, lon)
            'ulv_climate': Torch tensor of shape (nstep parameter, level, lat, lon)
            'lead_time': Integer.
            'input_time': Interger

        padding: Dictionary with keys 'level', 'lat', 'lon', each of dim 2.

    Returns:
        Dictionary with the following keys::

            'x': [batch, time, parameter, lat, lon]
            'ys': [batch, nsteps, parameter, lat, lon]
            'static': [batch, nstep, parameter, lat, lon]
            'lead_time': [batch]
            'input_time': [batch]
            'climate (Optional)': [batch, nsteps, parameter, lat, lon]

    Note:
        Here, for x and ys, 'parameter' is [surface parameter, upper level,
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

    lead_time = torch.empty(
        (nbatch, *b0["lead_time"].shape),
        dtype=torch.float32,
    )
    input_time = torch.empty((nbatch,), dtype=torch.float32)

    for i, rec in enumerate(batch):
        sur_x[i] = torch.Tensor(rec["sur_vals"])
        sur_y[i] = torch.Tensor(rec["sur_tars"])

        upl_x[i] = torch.Tensor(rec["ulv_vals"])
        upl_y[i] = torch.Tensor(rec["ulv_tars"])

        sur_sta[i] = torch.Tensor(rec["sur_static"])

        lead_time[i] = rec["lead_time"]
        input_time[i] = rec["input_time"]

    return_value = {
        "lead_time": lead_time,
        "input_time": input_time,
        "target_time": torch.sum(lead_time).reshape(-1),
    }

    # Reshape (batch, parameter, level, time, lat, lon)
    #   -> (batch, time, parameter, level, lat, lon)
    upl_x = upl_x.permute((0, 3, 1, 2, 4, 5))
    upl_y = upl_y.permute((0, 3, 1, 2, 4, 5))

    # Reshape (batch, parameter, time, lat, lon)
    #   -> (batch, time, parameter, lat, lon)
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
    return_value["statics"] = pad2d(sur_sta).contiguous()

    # We stack along the combined parameter level dimension
    return_value["x"] = torch.cat(
        (sur_x, upl_x.view(*upl_x.shape[:2], -1, *upl_x.shape[4:])), dim=2
    )
    return_value["ys"] = torch.cat(
        (sur_y, upl_y.view(*upl_y.shape[:2], -1, *upl_y.shape[4:])), dim=2
    )

    if climate_keys.issubset(data_keys):
        sur_climate = torch.empty((nbatch, *b0["sur_climate"].shape))
        ulv_climate = torch.empty((nbatch, *b0["ulv_climate"].shape))
        for i, rec in enumerate(batch):
            sur_climate[i] = rec["sur_climate"]
            ulv_climate[i] = rec["ulv_climate"]
        sur_climate = pad2d(sur_climate)
        ulv_climate = pad3d(ulv_climate)

        ulv_climate = ulv_climate.view(
            *ulv_climate.shape[:2], -1, *ulv_climate.shape[4:]
        )
        return_value["climates"] = torch.cat((sur_climate, ulv_climate), dim=2)

    return return_value


class RolloutSpec(SampleSpec):
    """
    A data class to collect the information used to define a rollout sample.
    """

    def __init__(
        self,
        inputs: tuple[pd.Timestamp, pd.Timestamp],
        lead_time: int,
        target: pd.Timestamp,
    ):
        """
        Args:
            inputs: Tuple of timestamps. In ascending order.
            lead_time: Lead time. In hours.
            target: Timestamp of the target. Can be before or after the inputs.
        """
        super().__init__(inputs, lead_time, target)

        self.dt = dt = pd.Timedelta(lead_time, unit="h")
        self.inters = list(pd.date_range(inputs[-1], target, freq=dt))

        self._ctimes = deepcopy(self.inters)
        self.stat_times = deepcopy(self.inters)

        self.stat_times.pop(-1)
        self._ctimes.pop(0)
        self.inters.pop(0)
        self.inters.pop(-1)

        self.times = [*inputs, *self.inters, target]
        self.targets = self.times[2:]
        self.nsteps = len(self.times) - 2

    @property
    def climatology_info(self) -> dict[pd.Timestamp, tuple[int, int]]:
        """Returns information required to obtain climatology data.
        Returns:
            list: list containing required climatology info.
        """
        return [(min(t.dayofyear, 365), t.hour) for t in self._ctimes]

    def _info_str(self) -> str:
        iso_8601 = "%Y-%m-%dT%H:%M:%S"

        inter_str = "\n".join(t.strftime(iso_8601) for t in self.inters)

        return (
            f"Issue time: {self.inputs[1].strftime(iso_8601)}\n"
            f"Lead time: {self.lead_time} hours ahead\n"
            f"Target time: {self.target.strftime(iso_8601)}\n"
            f"Intermediate times: {inter_str}"
        )

    @classmethod
    def get(cls, timestamp: pd.Timestamp, lead_time: int, nsteps: int):
        """Given a timestamp and lead time, generates a RolloutSpec object
        describing the sample further.

        Args:
            timestamp: Timstamp (issue time) of the sample.
            lead_time: Lead time. In hours.

        Returns:
            SampleSpec object.
        """
        if lead_time > 0:
            dt = pd.to_timedelta(lead_time, unit="h")
            timestamp_target = timestamp + nsteps * dt
        else:
            raise ValueError("Rollout is only forwards")

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


class Merra2RolloutDataset(Merra2Dataset):
    """Dataset class that read MERRA2 data for performing rollout.

    Implementation details::

        Samples stores the list of valid samples. This takes the form
        ```
        [
            [(timestamp 1, -input_time, n_steps)],
            [(timestamp 2, -input_time, n_steps)],
        ]
        ```
        The nested list is for compatibility reasons with Merra2Dataset. Note
        that input time and n_steps are always the same value. For some reason
        the sign of input_time is the opposite to that in Merra2Dataset
    """

    input_time_len = 2

    def __init__(
        self,
        time_range: tuple[str | pd.Timestamp, str | pd.Timestamp],
        input_time: int | float | pd.Timedelta,
        lead_time: int | float,
        data_path_surface: str | Path,
        data_path_vertical: str | Path,
        climatology_path_surface: str | Path | None,
        climatology_path_vertical: str | Path | None,
        surface_vars: list[str],
        static_surface_vars: list[str],
        vertical_vars: list[str],
        levels: list[float],
        roll_longitudes: int = 0,
        positional_encoding: str = "absolute",
    ):
        """
        Args:
            time_range: time range to consider when building dataset
            input_time: requested time between inputs
            lead_time: requested time to predict
            data_path_surface: path of surface data directory
            data_path_vertical: path of vertical data directory
            climatology_path_surface: path of surface climatology data
            directory
            climatology_path_vertical: path of vertical climatology data
            directory
            surface_vars: surface variables to return
            static_surface_vars: static surface variables to return
            vertical_vars: vertical variables to return
            levels: MERA2 vertical levels to consider
            roll_longitudes: Whether and now uch to randomly roll latitudes by.
            Defaults to 0.
            positional_encoding: The type of possitional encodeing to use.
            Defaults to "absolute".

        Raises:
            ValueError: If lead time is not integer multiple of input time
        """

        self._target_lead = lead_time

        if isinstance(input_time, int) or isinstance(input_time, float):
            self.timedelta_input = pd.to_timedelta(-input_time, unit="h")
        else:
            self.timedelta_input = -input_time

        lead_times = [self.timedelta_input / pd.to_timedelta(1, unit="h")]

        super().__init__(
            time_range,
            lead_times,
            [input_time],
            data_path_surface,
            data_path_vertical,
            climatology_path_surface,
            climatology_path_vertical,
            surface_vars,
            static_surface_vars,
            vertical_vars,
            levels,
            roll_longitudes,
            positional_encoding,
        )

        nstep_float = (
            pd.to_timedelta(self._target_lead, unit="h") / self.timedelta_input
        )

        if abs(nstep_float % 1) > 1e-5:
            raise ValueError("Leadtime not multiple of input time")

        self.nsteps = round(nstep_float)

    @ft.cached_property
    def samples(self) -> list[tuple[pd.Timestamp, int, int]]:
        """Generates list of all valid samlpes.

        Returns:
            List of tuples (timestamp, input time, lead time).
        """
        valid_samples = []

        for timestamp in sorted(self.valid_timestamps):
            timestamp_samples = []
            for lt in self.lead_times:
                spec = RolloutSpec.get(timestamp, lt, self.nsteps)

                if self._data_available(spec):
                    timestamp_samples.append(
                        (timestamp, self.input_times[0], lt, self.nsteps)
                    )

            if timestamp_samples:
                valid_samples.append(timestamp_samples)

        return valid_samples

    def get_data_from_rollout_spec(
        self, spec: RolloutSpec
    ) -> dict[str, Tensor | int | float]:
        """Loads and assembles sample data given a RolloutSpec object.

        Args:
            spec (RolloutSpec): Full details regarding the data to be loaded
        Returns:
            dict: Dictionary with keys 'sur_static', 'sur_vals', 'sur_tars',
            'ulv_vals', 'ulv_tars', 'sur_climate', 'ulv_climate',c'lead_time',
            'input_time'. For each, the value is as follows::

            {
                'sur_static': Torch tensor of shape [parameter, lat, lon]. For
                each pixel (lat, lon), the first 7 dimensions index sin(lat),
                cos(lon), sin(lon), cos(doy), sin(doy), cos(hod), sin(hod).
                Where doy is the day of the year [1, 366] and hod the hour of
                the day [0, 23].
                'sur_vals': Torch tensor of shape [parameter, time, lat, lon].
                'sur_tars': Torch tensor of shape [parameter, time, lat, lon].
                'ulv_vals': Torch tensor of shape
                [parameter, level, time, lat, lon].
                'ulv_tars': Torch tensor of shape
                [nsteps, parameter, level, time, lat, lon].
                'sur_climate': Torch tensor of shape
                [nsteps, parameter, lat, lon].
                'ulv_climate': Torch tensor of shape
                [nsteps, paramter, level, lat, lon].
                'lead_time': Float.
                'input_time': Float.
            }

        """

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

        # Load the static data
        stat = {}
        for data_files, times in stat_file_map.items():
            for time in times:
                hod, doy = time.hour, time.dayofyear
                stat[time] = self._read_static_data(data_files[0], doy, hod)

        # Combine times
        sample_data = {}

        input_upl = np.stack([data[t]["vert"] for t in spec.inputs], axis=2)
        sample_data["ulv_vals"] = input_upl

        target_upl = np.stack([data[t]["vert"] for t in spec.targets], axis=2)
        sample_data["ulv_tars"] = target_upl

        input_sur = np.stack([data[t]["surf"] for t in spec.inputs], axis=1)
        sample_data["sur_vals"] = input_sur

        target_sur = np.stack([data[t]["surf"] for t in spec.targets], axis=1)
        sample_data["sur_tars"] = target_sur

        # Load the static data
        static = np.stack([stat[t] for t in spec.stat_times], axis=0)
        sample_data["sur_static"] = static

        # If required load the climate data
        if self._require_clim:
            clim_data = {}
            for ci in spec.climatology_info:
                ci_year, ci_hour = ci

                surf_file = self.data_file_surface_climate(
                    dayofyear=ci_year,
                    hourofday=ci_hour,
                )

                vert_file = self.data_file_vertical_climate(
                    dayofyear=ci_year,
                    hourofday=ci_hour,
                )

                clim_data[ci] = self._read_climate((surf_file, vert_file))

            clim_surf = [clim_data[ci]["surf"] for ci in spec.climatology_info]
            sample_data["sur_climate"] = np.stack(clim_surf, axis=0)

            clim_surf = [clim_data[ci]["vert"] for ci in spec.climatology_info]
            sample_data["ulv_climate"] = np.stack(clim_surf, axis=0)

        # Move the data from numpy to torch
        sample_data = self._to_torch(sample_data, dtype=self.dtype)

        # Optionally roll
        if len(self._roll_longitudes) > 0:
            roll_by = random.choice(self._roll_longitudes)
            sample_data = self._lat_roll(sample_data, roll_by)

        # Now that we have rolled, we can add the static data
        lt = torch.tensor([spec.lead_time] * self.nsteps).to(self.dtype)
        sample_data["lead_time"] = lt
        sample_data["input_time"] = spec.input_time

        return sample_data

    def get_data(
        self, timestamp: pd.Timestamp, *args, **kwargs
    ) -> dict[Tensor | int]:
        """Loads data based on timestamp and lead time.

        Args:
            timestamp: Timestamp.
         Returns:
            Dictionary with keys 'sur_static', 'sur_vals', 'sur_tars',
              'ulv_vals', 'ulv_tars', 'sur_climate', 'ulv_climate',
              'lead_time', 'input_time'
        """
        rollout_spec = RolloutSpec.get(
            timestamp, self.lead_times[0], self.nsteps
        )
        sample_data = self.get_data_from_rollout_spec(rollout_spec)
        return sample_data
