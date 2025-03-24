from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

def get_data():
    pass

def get_model_data():
    surf_in_scal_path = Path("./climatology/musigma_surface.nc")
    hf_hub_download(
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        filename=f"climatology/{surf_in_scal_path.name}",
        local_dir="data",
    )

    vert_in_scal_path = Path("./climatology/musigma_vertical.nc")
    hf_hub_download(
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        filename=f"climatology/{vert_in_scal_path.name}",
        local_dir="data",
    )

    surf_out_scal_path = Path("./climatology/anomaly_variance_surface.nc")
    hf_hub_download(
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        filename=f"climatology/{surf_out_scal_path.name}",
        local_dir="data",
    )

    vert_out_scal_path = Path("./climatology/anomaly_variance_vertical.nc")
    hf_hub_download(
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        filename=f"climatology/{vert_out_scal_path.name}",
        local_dir="data",
    )
