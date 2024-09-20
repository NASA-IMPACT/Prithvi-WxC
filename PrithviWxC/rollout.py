import torch
from torch import Tensor, nn


def rollout_iter(
    nsteps: int,
    model: nn.Module,
    batch: dict[str, Tensor | int | float],
) -> Tensor:
    if nsteps < 1:
        raise ValueError("'nsteps' shouold be a positive int.")

    xlast = batch["x"][:, 1]
    batch["lead_time"] = batch["lead_time"][..., 0]
    batch["ys"] = batch["y"]

    # Save the masking ratio to be restored later
    mask_ratio_tmp = model.mask_ratio_inputs

    for step in range(nsteps):
        # After first step, turn off masking
        if step > 0:
            model.mask_ratio_inputs = 0.0

        batch["static"] = batch["statics"][:, step]
        batch["climate"] = batch["climates"][:, step]
        batch["y"] = batch["ys"][:, step]

        out = model(batch)

        batch["x"] = torch.cat((xlast[:, None], out[:, None]), dim=1)
        xlast = out

    # Restore the masking ratio
    model.mask_ratio_inputs = mask_ratio_tmp

    return xlast