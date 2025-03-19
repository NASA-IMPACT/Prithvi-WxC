from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr


def generate_time_sequence(start_time, end_time):
    """Generate a sequence of timestamps at 3-hour intervals.

    Args:
        start_time (datetime): The starting timestamp
        end_time (datetime): The ending timestamp

    Returns:
        list[datetime]: List of datetime objects at 3-hour intervals from start_time to end_time
    """
    current_time = start_time
    time_sequence = []

    while current_time <= end_time:
        time_sequence.append(current_time)
        current_time += timedelta(hours=3)

    return time_sequence


def get_var_aggregate_timestamps(input_datetimes):
    """Generate timestamp windows for surface variable aggregation.

    For each input timestamp, creates a window of ±30 minutes around it for averaging.
    The first timestamp is handled specially with only a forward window.

    Args:
        input_datetimes (list[datetime]): List of target timestamps for aggregation

    Returns:
        dict[datetime, tuple[datetime, datetime]]: Dictionary mapping each input timestamp
            to a tuple of (window_start, window_end) timestamps
    """
    result_timestamps = {}

    if input_datetimes:
        result_timestamps[input_datetimes[0]] = (
            input_datetimes[0],
            input_datetimes[0] + timedelta(minutes=30),
        )

    for input_datetime in input_datetimes[1:]:
        datetime_before = input_datetime - timedelta(minutes=30)
        datetime_after = input_datetime + timedelta(minutes=30)
        result_timestamps[input_datetime] = (datetime_before, datetime_after)

    return result_timestamps


def aggregate_merra(merra_data, grouping_type="sfc"):
    """Aggregate MERRA-2 data to 3-hourly intervals.

    This function takes MERRA-2 data and aggregates it to 3-hourly intervals using
    different aggregation strategies based on the variable type. For surface variables,
    it uses a centered average around each 3-hour timestamp. For precipitation, it uses
    a backward-looking accumulation.

    Args:
        merra_data (xarray.Dataset): Input MERRA-2 dataset containing a time dimension
            with datetime64 values.
        grouping_type (str, optional): Type of aggregation to perform. Must be either
            'sfc' for surface variables or 'pcp' for precipitation. Defaults to "sfc".

    Returns:
        xarray.Dataset: Aggregated dataset with 3-hourly timestamps (00, 03, 06, etc).
            Each timestamp represents either:
            - For sfc: The mean of data ±30 minutes around the timestamp
            - For pcp: The accumulation over previous 1-3 hours before the timestamp

    Raises:
        ValueError: If grouping_type is not 'sfc' or 'pcp'.

    Example:
        >>> merra_hourly = xr.open_dataset('merra_hourly.nc')
        >>> merra_3hr = aggregate_merra(merra_hourly, grouping_type='sfc')
    """
    start_time = (
        pd.to_datetime(merra_data.time[0].values)
        .to_pydatetime()
        .replace(hour=0, minute=0)
    )
    end_time = (
        pd.to_datetime(merra_data.time[-1].values)
        .to_pydatetime()
        .replace(hour=23, minute=59)
    )

    if grouping_type == "sfc":
        var_time = generate_time_sequence(start_time, end_time)
        var_time_groups = get_var_aggregate_timestamps(var_time)
    elif grouping_type == "pcp":
        var_time = generate_time_sequence(start_time, end_time)
        var_time_groups = get_var_aggregate_timestamps(var_time)
    else:
        raise ValueError("Invalid value for grouping_type. Use 'sfc' or 'pcp'.")

    merra_data_3hrly = []

    for vt in var_time:
        start_time, end_time = (var_time_groups[vt][0], var_time_groups[vt][1])

        time_mask = (
            merra_data["time"].values.astype("datetime64[ns]")
            >= np.datetime64(start_time)
        ) & (
            merra_data["time"].values.astype("datetime64[ns]")
            <= np.datetime64(end_time)
        )

        merra_data_3hrly.append(merra_data.isel(time=time_mask).mean(dim="time"))

    merra_data_3hrly_combined = xr.concat(merra_data_3hrly, dim="time")
    merra_data_3hrly_aggregated = merra_data_3hrly_combined.assign_coords(
        time=np.array(list(var_time_groups.keys()))
    )
    return merra_data_3hrly_aggregated
