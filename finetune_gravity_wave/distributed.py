import os
import random
from datetime import timedelta
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed import checkpoint as dist_checkpoint
from torch.distributed import fsdp


class _MPI:
    def __getattr__(self, attr):
        from mpi4py import MPI

        return getattr(MPI, attr)


mpi = _MPI()


def init_dist(device: str, rank: int, world_size: int):
    torch.distributed.init_process_group(
        backend=device,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=600),
    )


def print0(*inp):
    if is_main_process():
        print(*inp)


def _get_local_rank():
    envs = [
        "MV2_COMM_WORLD_LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "LOCAL_RANK",
    ]

    for ev in envs:
        if ev in os.environ:
            return int(os.environ[ev])
    else:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED).rank


def _get_rank():
    try:
        return int(os.environ["RANK"])
    except KeyError:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm.rank


def _get_world_size():
    try:
        return int(os.environ["WORLD_SIZE"])
    except KeyError:
        from mpi4py import MPI

        return MPI.COMM_WORLD.size


def init_ddp(use_gpu: bool):
    local_rank = _get_local_rank()
    rank = _get_rank()
    world_size = _get_world_size()

    if use_gpu:
        assert (
            torch.cuda.is_available()
        ), "GPU requested but none was found in the system."

    if use_gpu:
        init_dist("nccl", rank, world_size)
        torch.cuda.set_device(local_rank)
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        cudnn.benchmark = True
    else:
        init_dist("gloo", rank, world_size)
    return local_rank, rank


def set_global_seed(rank):
    random.seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_local_world_size():
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size is None:
        return 1
    else:
        return int(local_world_size)


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_node_rank():
    node_rank = os.environ.get("GROUP_RANK")
    if node_rank is None:
        return 0
    else:
        return int(node_rank)


def get_local_rank():
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return 0
    else:
        return int(local_rank)


def is_main_process():
    return get_rank() == 0


def model_distribute(
    model: torch.nn.Module,
    unet_encoder=None,
    unet_decoder=None,
    use_gpu: bool = True,
):
    if not use_gpu:
        raise RuntimeError("For FSDP currently we require GPU support")

    wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=10_000,
    )
    fsdp_kwargs_encoder_decoder = {
        # Options:
        # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD, _HYBRID_SHARD_ZERO2
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": torch.cuda.current_device(),
        "use_orig_params": False,
        "auto_wrap_policy": wrap_policy,
        "mixed_precision": MixedPrecision(
            param_dtype=torch.bfloat16,
            # reduce_dtype=torch.bfloat16,
            # buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
    }
    fsdp_kwargs_unembed = {
        # Options:
        # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD, _HYBRID_SHARD_ZERO2
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": torch.cuda.current_device(),
        "use_orig_params": False,
        # "auto_wrap_policy": wrap_policy,
        "mixed_precision": MixedPrecision(
            param_dtype=torch.float32, cast_forward_inputs=True
        ),
    }
    fsdp_kwargs_model = {
        # Options:
        # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD, _HYBRID_SHARD_ZERO2
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": torch.cuda.current_device(),
        "use_orig_params": False,
        # "auto_wrap_policy": wrap_policy,
        "mixed_precision": MixedPrecision(
            param_dtype=torch.float32,
            # reduce_dtype=torch.float32,
            # buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        ),
    }

    device = torch.cuda.current_device()
    torch.cuda.empty_cache()
    model = model.to(device)
    model.encoder = FullyShardedDataParallel(
        model.encoder, **fsdp_kwargs_encoder_decoder
    )
    model.decoder = FullyShardedDataParallel(
        model.decoder, **fsdp_kwargs_encoder_decoder
    )
    model.unembed = FullyShardedDataParallel(model.unembed, **fsdp_kwargs_unembed)

    if unet_decoder or unet_decoder:
        unet_encoder = unet_encoder.to(device)
        unet_decoder = unet_decoder.to(device)

        unet_encoder = FullyShardedDataParallel(unet_encoder, **fsdp_kwargs_model)
        unet_decoder = FullyShardedDataParallel(unet_decoder, **fsdp_kwargs_model)
        return model, unet_encoder, unet_decoder

    return model


def save_model_singular(model, *args, **kwargs):
    """Stream all model parameters to rank 0 on the CPU, then pass all
    other given arguments to `torch.save` to save the model, but only on
    the root process.
    """
    save_policy = fsdp.FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with fsdp.FullyShardedDataParallel.state_dict_type(
        model,
        fsdp.StateDictType.FULL_STATE_DICT,
        save_policy,
    ):
        cpu_state = model.state_dict()
    if is_main_process():
        torch.save(cpu_state, *args, **kwargs)
