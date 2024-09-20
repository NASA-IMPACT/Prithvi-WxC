from yacs.config import CfgNode as CN

_CN = CN()

_CN.wandb_mode = "disabled"
_CN.vartype = "uvtp122"
_CN.train_data_path = (
    "/nobackupnfs1/sroy14/rawdata/downstream_data/gravity_wave_flux/uvtp122"
)
_CN.valid_data_path = (
    "/nobackupnfs1/sroy14/rawdata/downstream_data/gravity_wave_flux/uvtp122/test"
)
_CN.singular_sharded_checkpoint = (
    "/nobackupnfs1/sroy14/checkpoints/prithvi_wxc/v0.8.50.rollout_step3.1.pth"
)
_CN.file_glob_pattern = "wxc_input_u_v_t_p_output_theta_uw_vw_era5_*.nc"

_CN.lr = 0.0001
_CN.hidden_channels = 160
_CN.n_lats_px = 64
_CN.n_lons_px = 128
_CN.in_channels_static = 3
_CN.mask_unit_size_px = [8, 16]
_CN.patch_size_px = [1, 1]


### Training Params

_CN.max_epochs = 100
_CN.batch_size = 12
_CN.num_data_workers = 8


def get_cfg():
    return _CN.clone()
