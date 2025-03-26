from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

def get_data():
    """
    We obtain data for ['2020-01-01T00:00:00', '2020-01-06T06:00:00']
    """
    surf_dir = Path("./merra-2")
    snapshot_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        allow_patterns="merra-2/MERRA2_sfc_2020010[1-6].nc",
        local_dir="data",
    )

    vert_dir = Path("./merra-2")
    snapshot_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        allow_patterns="merra-2/MERRA_pres_2020010[1-6].nc",
        local_dir="data",
    )

    surf_clim_dir = Path("./climatology")
    snapshot_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        allow_patterns="climatology/climate_surface_doy00[1-6]*.nc",
        local_dir="data",
    )

    vert_clim_dir = Path("./climatology")
    snapshot_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        allow_patterns="climatology/climate_vertical_doy00[1-6]*.nc",
        local_dir="data",
    )

def get_model_data():
    """
    We are getting the model data for the rollout model.
    """
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M-rollout",
        filename="config.yaml",
        local_dir="data",
    )
    
    weights_path = Path("./weights/prithvi.wxc.rollout.2300m.v1.pt")
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M-rollout",
        filename=weights_path.name,
        local_dir="data/weights",
    )

    surf_in_scal_path = Path("./climatology/musigma_surface.nc")
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=f"climatology/{surf_in_scal_path.name}",
        local_dir="data",
    )

    vert_in_scal_path = Path("./climatology/musigma_vertical.nc")
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=f"climatology/{vert_in_scal_path.name}",
        local_dir="data",
    )

    surf_out_scal_path = Path("./climatology/anomaly_variance_surface.nc")
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=f"climatology/{surf_out_scal_path.name}",
        local_dir="data",
    )

    vert_out_scal_path = Path("./climatology/anomaly_variance_vertical.nc")
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=f"climatology/{vert_out_scal_path.name}",
        local_dir="data",
    )