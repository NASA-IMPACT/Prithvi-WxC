from importlib.metadata import version
TORCH_VERSION = version('torch')

from functools import cached_property
from typing import Optional
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
if TORCH_VERSION > '2.3.0':
    from torch.nn.attention import SDPBackend, sdpa_kernel


# DropPath code is straight from timm
# (https://huggingface.co/spaces/Roll20/pet_score/blame/main/lib/timm/models/layers/drop.py)
# Primarily since we currently don't have timm in the environment.
def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    """
    Multi layer perceptron.
    """

    def __init__(
        self, features: int, hidden_features: int, dropout: float = 0.0
    ) -> None:
        """
        Args:
            features: Input/output dimension.
            hidden_features: Hidden dimension.
            dropout: Dropout.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, features),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            Tensor of shape [..., channel]
        Returns:
            Tensor of same shape as x.
        """
        return self.net(x)


class LayerNormPassThrough(nn.LayerNorm):
    """
    Normalising layer that allows the attention mask to be passed through
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, d: tuple[Tensor, Tensor | None]) -> tuple[Tensor, Tensor | None]:
        """
        Forwards function
        Args:
            d: tuple of the data tensor and the attention mask
        Returns:
            output: normalised output data
            attn_mask: the attention mask that was passed in
        """
        input, attn_mask = d
        output = F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )
        return output, attn_mask


class MultiheadAttention(nn.Module):
    """
    Multihead attention layer for inputs of shape [..., sequence, features].

    Uses `scaled_dot_product_attention` to obtain a memory efficient attention
    computation (https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
    This follows:
    - Dao et la. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
        (https://arxiv.org/abs/2205.14135)
    - Rabe, Staats "Self-attention Does Not Need O(n2) Memory" (https://arxiv.org/abs/2112.05682)

    Note: Even though the documentation page for `scaled_dot_product_attention`
    states that tensors can have any number of dimensions as long as the shapes
    are along the lines of `(B, ..., S, E)`, the fused and memory efficient
    mechanisms we enforce here require a 4D input. Some experimentatino shows
    that this should be of shape `(B, H, S, E)`, where `H` represents heads.
    However, as of right now this is not confirmed int he documentation.
    """

    def __init__(self, features: int, n_heads: int, dropout: float) -> None:
        """
        Args:
            features: Number of features for inputs to the layer.
            n_heads: Number of attention heads. Should be a factor of features.
                (I.e. the layer uses features // n_heads.)
            dropout: Dropout.
        """
        super().__init__()

        if not (features % n_heads) == 0:
            raise ValueError(
                f"Number of features {features} is not divisible by number of heads {n_heads}."
            )

        self.features = features
        self.n_heads = n_heads
        self.dropout = dropout

        self.qkv_layer = torch.nn.Linear(features, features * 3, bias=False)
        self.w_layer = torch.nn.Linear(features, features, bias=False)

    def forward(self, d: tuple[Tensor, Tensor | None]) -> Tensor:
        """
        Args:
            d: tuple containing Tensor of shape [..., sequence, features] and
            the attention mask
        Returns:
            Tensor of shape [..., sequence, features]
        """
        x, attn_mask = d

        if not x.shape[-1] == self.features:
            raise ValueError(
                f"Expecting tensor with last dimension of size {self.features}."
            )

        passenger_dims = x.shape[:-2]
        B = passenger_dims.numel()
        S = x.shape[-2]
        C = x.shape[-1]
        x = x.reshape(B, S, C)

        # x [B, S, C]
        # q, k, v [B, H, S, C/H]
        q, k, v = (
            self.qkv_layer(x)
            .view(B, S, self.n_heads, 3 * (C // self.n_heads))
            .transpose(1, 2)
            .chunk(chunks=3, dim=3)
        )

        # Let us enforce either flash (A100+) or memory efficient attention.
        if TORCH_VERSION > '2.3.0':
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                # x [B, H, S, C//H]
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout
                )
        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                # x [B, H, S, C//H]
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)

        # x [B, S, C]
        x = x.transpose(1, 2).view(B, S, C)

        # x [B, S, C]
        x = self.w_layer(x)

        # Back to input shape
        x = x.view(*passenger_dims, S, self.features)
        return x


class Transformer(nn.Module):
    """
    Transformer for inputs of shape [..., S, features].
    """

    def __init__(
        self,
        features: int,
        mlp_multiplier: int,
        n_heads: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        """
        Args:
            features: Number of features for inputs to the layer.
            mlp_multiplier: Model will use features*mlp_multiplier hidden units.
            n_heads: Number of attention heads. Should be a factor of features.
                (I.e. the layer uses features // n_heads.)
            dropout: Dropout.
            drop_path: DropPath.
        """
        super().__init__()

        self.features = features
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.attention = nn.Sequential(
            LayerNormPassThrough(features),
            MultiheadAttention(features, n_heads, dropout),
        )

        self.ff = nn.Sequential(
            nn.LayerNorm(features),
            Mlp(
                features=features,
                hidden_features=features * mlp_multiplier,
                dropout=dropout,
            ),
        )

    def forward(self, d: tuple[Tensor, Tensor | None]) -> Tensor:
        """
        Args:
            x: Tensor of shape [..., sequence, features]
        Returns:
            Tensor of shape [..., sequence, features]
        """
        x, attn_mask = d
        if not x.shape[-1] == self.features:
            raise ValueError(
                f"Expecting tensor with last dimension of size {self.features}."
            )

        attention_x = self.attention(d)

        x = x + self.drop_path(attention_x)
        x = x + self.drop_path(self.ff(x))

        return x

class _Shift(nn.Module):
    """
    Private base class for the shifter. This allows some behaviour to be easily
    handled when the shifter isn't used.
    """

    def __init__(self):
        super().__init__()

        self._shifted = False

    @torch.no_grad()
    def reset(self) -> None:
        """
        Resets the bool tracking whether the data is shifted
        """
        self._shifted: bool = False

    def forward(self, data: Tensor) -> tuple[Tensor, dict[bool, None]]:
        return data, {True: None, False: None}


class SWINShift(_Shift):
    """
    Handles the shifting of patches similar to how SWIN works. However if we
    shift the latitudes then the poles will wrap and potentially that might be
    problematic. The possition tokens should handle it but masking is safer.
    """

    def __init__(
        self,
        mu_shape: tuple[int, int],
        global_shape: tuple[int, int],
        local_shape: tuple[int, int],
        patch_shape: tuple[int, int],
        n_context_tokens: int = 2,
    ) -> None:
        """
        Args:
            mu_shape: the shape to the masking units
            global_shape: number of global patches in lat and lon
            local_shape: size of the local patches
            patch_shape: patch size
            n_context_token: number of additional context tokens at start of _each_ local sequence
        """
        super().__init__()

        self._mu_shape = ms = mu_shape
        self._g_shape = gs = global_shape
        self._l_shape = ls = local_shape
        self._p_shape = ps = patch_shape
        self._lat_patch = (gs[0], ls[0], gs[1], ls[1])
        self._n_context_tokens = n_context_tokens

        self._g_shift_to = tuple(int(0.5 * x / p) for x, p in zip(ms, ps))
        self._g_shift_from = tuple(-int(0.5 * x / p) for x, p in zip(ms, ps))

        # Define the attention masks for the shifted MaxViT.
        nglobal = global_shape[0] * global_shape[1]
        nlocal = local_shape[0] * local_shape[1] + self._n_context_tokens  # "+ 1" for leadtime

        lm = torch.ones((nglobal, 1, nlocal, nlocal), dtype=bool)
        mwidth = int(0.5 * local_shape[1]) * local_shape[0]
        lm[
            : gs[1],
            :,
            self._n_context_tokens : mwidth + self._n_context_tokens,
            self._n_context_tokens : mwidth + self._n_context_tokens,
        ] = False
        self.register_buffer("local_mask", lm)

        gm = torch.ones((nlocal, 1, nglobal, nglobal), dtype=bool)
        gm[: int(0.5 * ls[1]) * ls[0], :, : gs[1], : gs[1]] = False
        self.register_buffer("global_mask", gm)

    def _to_grid_global(self, x: Tensor) -> Tensor:
        """
        Shuffle and reshape the data from the global/local setting back to the
        lat/lon grid setting
        Args:
            x: the data tensor to be shuffled.
        Returns:
            x: data in the global/local setting
        """
        nbatch, *other = x.shape

        y1 = x.view(nbatch, *self._g_shape, *self._l_shape, -1)
        y2 = y1.permute(0, 5, 1, 3, 2, 4).contiguous()

        s = y2.shape
        return y2.view((nbatch, -1, s[2] * s[3], s[4] * s[5]))

    def _to_grid_local(self, x: Tensor) -> Tensor:
        """
        Shuffle and reshape the data from the local/global setting to the
        lat/lon grid setting
        Args:
            x: the data tensor to be shuffled.
        Returns:
            x: data in the lat/lon setting.
        """
        x = x.transpose(2, 1).contiguous()
        return self._to_grid_global(x)

    def _from_grid_global(self, x: Tensor) -> Tensor:
        """
        Shuffle and reshape the data from the lat/lon grid to the global/local
        setting
        Args:
            x: the data tensor to be shuffled.
        Returns:
            x: data in the global/local setting
        """
        nbatch, *other = x.shape

        z1 = x.view(nbatch, -1, *self._lat_patch)
        z2 = z1.permute(0, 2, 4, 3, 5, 1).contiguous()

        s = z2.shape
        return z2.view(nbatch, s[1] * s[2], s[3] * s[4], -1)

    def _from_grid_local(self, x: Tensor) -> Tensor:
        """
        Shuffle and reshape the data from the lat/lon grid to the local/global
        setting
        Args:
            x: the data tensor to be shuffled.
        Returns:
            x: data in the local/global setting
        """
        x = self._from_grid_global(x)
        return x.transpose(2, 1).contiguous()

    def _shift(self, x: Tensor) -> Tensor:
        """
        Shifts data in the gridded lat/lon setting by half the mask unit shape
        Args:
            x: data to be shifted
        Returns:
            x: either the hsifted or unshifted data
        """
        shift = self._g_shift_from if self._shifted else self._g_shift_to
        x_shifted = torch.roll(x, shift, (-2, -1))

        self._shifted = not self._shifted
        return x_shifted

    def _sep_lt(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Seperate off the leadtime from the local patches
        Args:
            x: data to have leadtime removed from
        Returns:
            lt: leadtime
            x: data without the lead time in the local patch
        """
        lt_it = x[:, : self._n_context_tokens, :, :]
        x_stripped = x[:, self._n_context_tokens :, :, :]

        return lt_it, x_stripped

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor]:
        """
        Shift or unshift the the data depending on whether the data is already
        shifted, as defined by self._shifted
        Args:
            data: data to be shifted
        Returns:

        """
        lt, x = self._sep_lt(data)

        x_grid = self._to_grid_local(x)
        x_shifted = self._shift(x_grid)
        x_patched = self._from_grid_local(x_shifted)

        # Mask has to be repeated based on batch size
        n_batch = x_grid.shape[0]
        local_rep = [n_batch] + [1] * (self.local_mask.ndim - 1)
        global_rep = [n_batch] + [1] * (self.global_mask.ndim - 1)

        if self._shifted:
            attn_mask = {
                True: self.local_mask.repeat(local_rep),
                False: self.global_mask.repeat(global_rep),
            }
        else:
            attn_mask = {True: None, False: None}

        return torch.cat((lt, x_patched), axis=1), attn_mask

class SWINShiftNoBuffer(_Shift):
    """
    Handles the shifting of patches similar to how SWIN works. However if we
    shift the latitudes then the poles will wrap and potentially that might be
    problematic. The possition tokens should handle it but masking is safer.
    """

    def __init__(
        self,
        mu_shape: tuple[int, int],
        global_shape: tuple[int, int],
        local_shape: tuple[int, int],
        patch_shape: tuple[int, int],
        n_context_tokens: int = 2,
    ) -> None:
        """
        Args:
            mu_shape: the shape to the masking units
            global_shape: number of global patches in lat and lon
            local_shape: size of the local patches
            patch_shape: patch size
            n_context_token: number of additional context tokens at start of _each_ local sequence
        """
        super().__init__()

        self._mu_shape = ms = mu_shape
        self._g_shape = gs = global_shape
        self._l_shape = ls = local_shape
        self._p_shape = ps = patch_shape
        self._lat_patch = (gs[0], ls[0], gs[1], ls[1])
        self._n_context_tokens = n_context_tokens

        self._g_shift_to = tuple(int(0.5 * x / p) for x, p in zip(ms, ps))
        self._g_shift_from = tuple(-int(0.5 * x / p) for x, p in zip(ms, ps))

        # Define the attention masks for the shifted MaxViT.
        nglobal = global_shape[0] * global_shape[1]
        nlocal = local_shape[0] * local_shape[1] + self._n_context_tokens  # "+ 1" for leadtime

        lm = torch.ones((nglobal, 1, nlocal, nlocal), dtype=bool)
        mwidth = int(0.5 * local_shape[1]) * local_shape[0]
        lm[
            : gs[1],
            :,
            self._n_context_tokens : mwidth + self._n_context_tokens,
            self._n_context_tokens : mwidth + self._n_context_tokens,
        ] = False
        self.local_mask = lm

        gm = torch.ones((nlocal, 1, nglobal, nglobal), dtype=bool)
        gm[: int(0.5 * ls[1]) * ls[0], :, : gs[1], : gs[1]] = False
        self.global_mask = gm

    def _to_grid_global(self, x: Tensor) -> Tensor:
        """
        Shuffle and reshape the data from the global/local setting back to the
        lat/lon grid setting
        Args:
            x: the data tensor to be shuffled.
        Returns:
            x: data in the global/local setting
        """
        nbatch, *other = x.shape

        y1 = x.view(nbatch, *self._g_shape, *self._l_shape, -1)
        y2 = y1.permute(0, 5, 1, 3, 2, 4).contiguous()

        s = y2.shape
        return y2.view((nbatch, -1, s[2] * s[3], s[4] * s[5]))

    def _to_grid_local(self, x: Tensor) -> Tensor:
        """
        Shuffle and reshape the data from the local/global setting to the
        lat/lon grid setting
        Args:
            x: the data tensor to be shuffled.
        Returns:
            x: data in the lat/lon setting.
        """
        x = x.transpose(2, 1).contiguous()
        return self._to_grid_global(x)

    def _from_grid_global(self, x: Tensor) -> Tensor:
        """
        Shuffle and reshape the data from the lat/lon grid to the global/local
        setting
        Args:
            x: the data tensor to be shuffled.
        Returns:
            x: data in the global/local setting
        """
        nbatch, *other = x.shape

        z1 = x.view(nbatch, -1, *self._lat_patch)
        z2 = z1.permute(0, 2, 4, 3, 5, 1).contiguous()

        s = z2.shape
        return z2.view(nbatch, s[1] * s[2], s[3] * s[4], -1)

    def _from_grid_local(self, x: Tensor) -> Tensor:
        """
        Shuffle and reshape the data from the lat/lon grid to the local/global
        setting
        Args:
            x: the data tensor to be shuffled.
        Returns:
            x: data in the local/global setting
        """
        x = self._from_grid_global(x)
        return x.transpose(2, 1).contiguous()

    def _shift(self, x: Tensor) -> Tensor:
        """
        Shifts data in the gridded lat/lon setting by half the mask unit shape
        Args:
            x: data to be shifted
        Returns:
            x: either the hsifted or unshifted data
        """
        shift = self._g_shift_from if self._shifted else self._g_shift_to
        x_shifted = torch.roll(x, shift, (-2, -1))

        self._shifted = not self._shifted
        return x_shifted

    def _sep_lt(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Seperate off the leadtime from the local patches
        Args:
            x: data to have leadtime removed from
        Returns:
            lt: leadtime
            x: data without the lead time in the local patch
        """
        lt_it = x[:, : self._n_context_tokens, :, :]
        x_stripped = x[:, self._n_context_tokens :, :, :]

        return lt_it, x_stripped

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor]:
        """
        Shift or unshift the the data depending on whether the data is already
        shifted, as defined by self._shifted
        Args:
            data: data to be shifted
        Returns:

        """
        lt, x = self._sep_lt(data)

        if self.local_mask.device != x.device:
            self.local_mask = self.local_mask.to(device=x.device)
        if self.global_mask.device != x.device:
            self.global_mask = self.global_mask.to(device=x.device)

        x_grid = self._to_grid_local(x)
        x_shifted = self._shift(x_grid)
        x_patched = self._from_grid_local(x_shifted)

        # Mask has to be repeated based on batch size
        n_batch = x_grid.shape[0]
        local_rep = [n_batch] + [1] * (self.local_mask.ndim - 1)
        global_rep = [n_batch] + [1] * (self.global_mask.ndim - 1)

        if self._shifted:
            attn_mask = {
                True: self.local_mask.repeat(local_rep),
                False: self.global_mask.repeat(global_rep),
            }
        else:
            attn_mask = {True: None, False: None}

        return torch.cat((lt, x_patched), axis=1), attn_mask

class LocalGlobalLocalBlock(nn.Module):
    """
    Applies alternating block and grid attention. Given a parameter n_blocks, the entire
    module contains 2*n_blocks+1 transformer blocks. The first, third, ..., last apply
    local (block) attention. The second, fourth, ... global (grid) attention.

    This is heavily inspired by Tu et al. "MaxViT: Multi-Axis Vision Transformer"
    (https://arxiv.org/abs/2204.01697).
    """

    def __init__(
        self,
        features: int,
        mlp_multiplier: int,
        n_heads: int,
        dropout: float,
        n_blocks: int,
        drop_path: float,
        shifter: nn.Module | None = None,
        checkpoint: list[int]=[],
    ) -> None:
        """
        Args:
            features: Number of features for inputs to the layer.
            mlp_multiplier: Model will use features*mlp_multiplier hidden units.
            n_heads: Number of attention heads. Should be a factor of features.
            (I.e. the layer uses features // n_heads.)
            dropout: Dropout.
            drop_path: DropPath.
            n_blocks: Number of local-global transformer pairs.
        """
        super().__init__()

        self.features = features
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = drop_path
        self.n_blocks = n_blocks
        self._checkpoint = checkpoint

        if len(checkpoint) > 0:
            if min(checkpoint) < 0 or max(checkpoint) >= 2 * n_blocks + 1:
                raise ValueError(f'Checkpoints should satisfy 0 <= i < 2*n_blocks+1. We have {checkpoint}.')

        self.transformers = nn.ModuleList(
            [
                Transformer(
                    features=features,
                    mlp_multiplier=mlp_multiplier,
                    n_heads=n_heads,
                    dropout=dropout,
                    drop_path=drop_path,
                )
                for _ in range(2 * n_blocks + 1)
            ]
        )

        self.evaluator = [
            self._checkpoint_wrapper if i in checkpoint else lambda m, x : m(x)
            for i, _ in enumerate(self.transformers)
        ]

        self.shifter = shifter or _Shift()

    @staticmethod
    def _checkpoint_wrapper(model, data):
        return checkpoint(model, data, use_reentrant=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch, global_sequence, local_sequence, features]
        Returns:
            Tensor of shape [batch, global_sequence, local_sequence, features]
        """
        if x.shape[-1] != self.features:
            raise ValueError(
                f"Expecting tensor with last dimension of size {self.features}."
            )
        if x.ndim != 4:
            raise ValueError(
                f"Expecting tensor with exactly four dimensions. Input has shape {x.shape}."
            )

        self.shifter.reset()
        local: bool = True
        attn_mask = {True: None, False: None}

        transformer_iter = zip(self.evaluator, self.transformers)

        # First local block
        evaluator, transformer = next(transformer_iter)
        x = evaluator(transformer, (x, attn_mask[local]))

        for evaluator, transformer in transformer_iter:
            local = not local
            # We are making exactly 2*n_blocks transposes.
            # So the output has the same shape as input.
            x = x.transpose(1, 2)

            x = evaluator(transformer, (x, attn_mask[local]))

            if not local:
                x, attn_mask = self.shifter(x)

        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding via 2D convolution.
    """

    def __init__(
        self, patch_size: int | tuple[int, ...], channels: int, embed_dim: int
    ):
        super().__init__()

        self.patch_size = patch_size
        self.channels = channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch, channels, lat, lon].
        Returns:
            Tensor with shape [batch, embed_dim, lat//patch_size, lon//patch_size]
        """

        H, W = x.shape[-2:]

        if W % self.patch_size[1] != 0:
            raise ValueError(
                f"Cannot do patch embedding for tensor of shape {x.size()}"
                " with patch size {self.patch_size}. (Dimensions are BSCHW.)"
            )
        if H % self.patch_size[0] != 0:
            raise ValueError(
                f"Cannot do patch embedding for tensor of shape {x.size()}"
                f" with patch size {self.patch_size}. (Dimensions are BSCHW.)"
            )

        x = self.proj(x)

        return x


class PrithviWxCEncoderDecoder(nn.Module):
    """
    Hiera-MaxViT encoder/decoder code.
    """

    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        mlp_multiplier: float,
        n_heads: int,
        dropout: float,
        drop_path: float,
        shifter: nn.Module | None = None,
        transformer_cp: list[int]=[],
    ) -> None:
        """
        Args:
            embed_dim: Embedding dimension
            n_blocks: Number of local-global transformer pairs.
            mlp_multiplier: MLP multiplier for hidden features in feed forward
                networks.
            n_heads: Number of attention heads.
            dropout: Dropout.
            drop_path: DropPath.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_blocks = n_blocks
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self._transformer_cp = transformer_cp

        self.lgl_block = LocalGlobalLocalBlock(
            features=embed_dim,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
            n_blocks=n_blocks,
            shifter=shifter,
            checkpoint=transformer_cp,
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, global sequence, local sequence, embed_dim]
        Returns:
            Tensor of shape [batch, mask_unit_sequence, local_sequence, embed_dim].
            Identical in shape to the input x.
        """

        x = self.lgl_block(x)

        return x


class PrithviWxC(nn.Module):
    """
    Encoder-decoder fusing Hiera with MaxViT. See
    - Ryali et al. "Hiera: A Hierarchical Vision Transformer without the
        Bells-and-Whistles" (https://arxiv.org/abs/2306.00989)
    - Tu et al. "MaxViT: Multi-Axis Vision Transformer"
        (https://arxiv.org/abs/2204.01697)
    """

    def __init__(
        self,
        in_channels: int,
        input_size_time: int,
        in_channels_static: int,
        input_scalers_mu: Tensor,
        input_scalers_sigma: Tensor,
        input_scalers_epsilon: float,
        static_input_scalers_mu: Tensor,
        static_input_scalers_sigma: Tensor,
        static_input_scalers_epsilon: float,
        output_scalers: Tensor,
        n_lats_px: int,
        n_lons_px: int,
        patch_size_px: tuple[int],
        mask_unit_size_px: tuple[int],
        mask_ratio_inputs: float,
        mask_ratio_targets: float,
        embed_dim: int,
        n_blocks_encoder: int,
        n_blocks_decoder: int,
        mlp_multiplier: float,
        n_heads: int,
        dropout: float,
        drop_path: float,
        parameter_dropout: float,
        residual: str,
        masking_mode: str,
        positional_encoding: str,
        encoder_shifting: bool = False,
        decoder_shifting: bool = False,
        checkpoint_encoder: list[int]=[],
        checkpoint_decoder: list[int]=[],
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            input_size_time: Number of timestamps in input.
            in_channels_static: Number of input channels for static data.
            input_scalers_mu: Tensor of size (in_channels,). Used to rescale
                input.
            input_scalers_sigma: Tensor of size (in_channels,). Used to rescale
                input.
            input_scalers_epsilon: Float. Used to rescale input.
            static_input_scalers_mu: Tensor of size (in_channels_static). Used
                to rescale static inputs.
            static_input_scalers_sigma: Tensor of size (in_channels_static).
                Used to rescale static inputs.
            static_input_scalers_epsilon: Float. Used to rescale static inputs.
            output_scalers: Tensor of shape (in_channels,). Used to rescale
                output.
            n_lats_px: Total latitudes in data. In pixels.
            n_lons_px: Total longitudes in data. In pixels.
            patch_size_px: Patch size for tokenization. In pixels lat/lon.
            mask_unit_size_px: Size of each mask unit. In pixels lat/lon.
            mask_ratio_inputs: Masking ratio for inputs. 0 to 1.
            mask_ratio_targets: Masking ratio for targets. 0 to 1.
            embed_dim: Embedding dimension
            n_blocks_encoder: Number of local-global transformer pairs in
                encoder.
            n_blocks_decoder: Number of local-global transformer pairs in
                decoder.
            mlp_multiplier: MLP multiplier for hidden features in feed forward
                networks.
            n_heads: Number of attention heads.
            dropout: Dropout.
            drop_path: DropPath.
            parameter_dropout: Dropout applied to parameters.
            residual: Indicates whether and how model should work as residual
                model. Accepted values are 'climate', 'temporal' and 'none'
            positional_encoding: possible values are ['absolute' (default), 'fourier'].
                'absolute'  lat lon encoded in 3 dimensions using sine and cosine
                'fourier' lat/lon to be encoded using various frequencies
            masking_mode: String ['local', 'global', 'both'] that controls the
                type of masking used.
            checkpoint_encoder: List of integers controlling if gradient checkpointing is used on encoder.
                Format: [] for no gradient checkpointing. [3, 7] for checkpointing after 4th and 8th layer etc.
            checkpoint_decoder: List of integers controlling if gradient checkpointing is used on decoder.
                Format: See `checkpoint_encoder`.
            masking_mode: The type of masking to use {'global', 'local', 'both'}
            encoder_shifting: Whether to use swin shifting in the encoder.
            decoder_shifting: Whether to use swin shifting in the decoder.
        """
        super().__init__()

        if mask_ratio_targets > 0.0:
            raise NotImplementedError("Target masking is not implemented.")

        self.in_channels = in_channels
        self.input_size_time = input_size_time
        self.in_channels_static = in_channels_static
        self.n_lats_px = n_lats_px
        self.n_lons_px = n_lons_px
        self.patch_size_px = patch_size_px
        self.mask_unit_size_px = mask_unit_size_px
        self.mask_ratio_inputs = mask_ratio_inputs
        self.mask_ratio_targets = mask_ratio_targets
        self.embed_dim = embed_dim
        self.n_blocks_encoder = n_blocks_encoder
        self.n_blocks_decoder = n_blocks_decoder
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = drop_path
        self.residual = residual
        self._encoder_shift = encoder_shifting
        self._decoder_shift = decoder_shifting
        self.positional_encoding = positional_encoding
        self._checkpoint_encoder = checkpoint_encoder
        self._checkpoint_decoder = checkpoint_decoder

        assert self.n_lats_px % self.mask_unit_size_px[0] == 0
        assert self.n_lons_px % self.mask_unit_size_px[1] == 0
        assert self.mask_unit_size_px[0] % self.patch_size_px[0] == 0
        assert self.mask_unit_size_px[1] % self.patch_size_px[1] == 0

        if self.patch_size_px[0] != self.patch_size_px[1]:
            raise NotImplementedError(
                "Current pixel shuffle implementation assumes same patch size along both dimensions."
            )

        self.local_shape_mu = (
            self.mask_unit_size_px[0] // self.patch_size_px[0],
            self.mask_unit_size_px[1] // self.patch_size_px[1],
        )
        self.global_shape_mu = (
            self.n_lats_px // self.mask_unit_size_px[0],
            self.n_lons_px // self.mask_unit_size_px[1],
        )

        assert input_scalers_mu.shape == (in_channels,)
        assert input_scalers_sigma.shape == (in_channels,)
        assert output_scalers.shape == (in_channels,)

        if self.positional_encoding != 'fourier':
            assert static_input_scalers_mu.shape == (in_channels_static,)
            assert static_input_scalers_sigma.shape == (in_channels_static,)

        # Input shape [batch, time, parameter, lat, lon]
        self.input_scalers_epsilon = input_scalers_epsilon
        self.register_buffer('input_scalers_mu', input_scalers_mu.reshape(1, 1, -1, 1, 1))
        self.register_buffer('input_scalers_sigma', input_scalers_sigma.reshape(1, 1, -1, 1, 1))

        # Static inputs shape [batch, parameter, lat, lon]
        self.static_input_scalers_epsilon = static_input_scalers_epsilon
        self.register_buffer('static_input_scalers_mu', static_input_scalers_mu.reshape(1, -1, 1, 1))
        self.register_buffer('static_input_scalers_sigma', static_input_scalers_sigma.reshape(1, -1, 1, 1))

        # Output shape [batch, parameter, lat, lon]
        self.register_buffer('output_scalers', output_scalers.reshape(1, -1, 1, 1))

        self.parameter_dropout = nn.Dropout2d(p=parameter_dropout)

        self.patch_embedding = PatchEmbed(
            patch_size=patch_size_px,
            channels=in_channels * input_size_time,
            embed_dim=embed_dim,
        )

        if self.residual == "climate":
            self.patch_embedding_static = PatchEmbed(
                patch_size=patch_size_px,
                channels=in_channels + in_channels_static,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embedding_static = PatchEmbed(
                patch_size=patch_size_px,
                channels=in_channels_static,
                embed_dim=embed_dim,
            )

        self.input_time_embedding = nn.Linear(1, embed_dim//4, bias=True)
        self.lead_time_embedding = nn.Linear(1, embed_dim//4, bias=True)

        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, self.embed_dim))
        self._nglobal_mu = np.prod(self.global_shape_mu)
        self._global_idx = torch.arange(self._nglobal_mu)

        self._nlocal_mu = np.prod(self.local_shape_mu)
        self._local_idx = torch.arange(self._nlocal_mu)

        if self._encoder_shift:
            self.encoder_shifter = e_shifter = SWINShiftNoBuffer(
                self.mask_unit_size_px,
                self.global_shape_mu,
                self.local_shape_mu,
                self.patch_size_px,
                n_context_tokens=0,
            )
        else:
            self.encoder_shifter = e_shifter = None
        self.encoder = PrithviWxCEncoderDecoder(
            embed_dim=embed_dim,
            n_blocks=n_blocks_encoder,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
            shifter=e_shifter,
            transformer_cp=checkpoint_encoder,
        )

        if n_blocks_decoder != 0:
            if self._decoder_shift:
                self.decoder_shifter = d_shifter = SWINShift(
                    self.mask_unit_size_px,
                    self.global_shape_mu,
                    self.local_shape_mu,
                    self.patch_size_px,
                    n_context_tokens=0,
                )
            else:
                self.decoder_shifter = d_shifter = None

            self.decoder = PrithviWxCEncoderDecoder(
                embed_dim=embed_dim,
                n_blocks=n_blocks_decoder,
                mlp_multiplier=mlp_multiplier,
                n_heads=n_heads,
                dropout=dropout,
                drop_path=0.,
                shifter=d_shifter,
                transformer_cp=checkpoint_decoder,
            )

            self.unembed = nn.Linear(
                self.embed_dim,
                self.in_channels * self.patch_size_px[0] * self.patch_size_px[1],
                bias=True,
            )

        self.masking_mode = masking_mode.lower()
        match self.masking_mode:
            case "local":
                self.generate_mask = self._gen_mask_local
            case "global":
                self.generate_mask = self._gen_mask_global
            case "both":
                self._mask_both_local: bool = True
                self.generate_mask = self._gen_mask_both
            case _:
                raise ValueError(f"Masking mode '{masking_mode}' not supported")

    def swap_masking(self) -> None:
        if hasattr(self, '_mask_both_local'):
            self._mask_both_local = not self._mask_both_local

    @cached_property
    def n_masked_global(self):
        return int(self.mask_ratio_inputs * np.prod(self.global_shape_mu))

    @cached_property
    def n_masked_local(self):
        return int(self.mask_ratio_inputs * np.prod(self.local_shape_mu))

    @staticmethod
    def _shuffle_along_axis(a, axis):
        # https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
        idx = torch.argsort(input=torch.rand(*a.shape), dim=axis)
        return torch.gather(a, dim=axis, index=idx)

    def _gen_mask_local(self, sizes: tuple[int]) -> tuple[Tensor]:
        """
        Args:
            batch_size: Number of elements in batch
        Returns:
            Tuple of torch tensors. [indices masked, indices unmasked].
            Each of these is a tensor of shape (batch, global sequene)
        """
        # We identifies which indices (values) should be masked

        maskable_indices = self._local_idx.view(1, -1).expand(*sizes[:2], -1)

        maskable_indices = self._shuffle_along_axis(maskable_indices, 2)

        # `...` cannot be jit'd :-(
        indices_masked = maskable_indices[:, :, : self.n_masked_local]
        indices_unmasked = maskable_indices[:, :, self.n_masked_local :]

        return indices_masked, indices_unmasked

    def _gen_mask_global(self, sizes: tuple[int]) -> tuple[Tensor]:
        """
        Args:
            batch_size: Number of elements in batch
        Returns:
            Tuple of torch tensors. [indices masked, indices unmasked].
            Each of these is a tensor of shape (batch, global sequene)
        """
        # We identifies which indices (values) should be masked

        maskable_indices = self._global_idx.view(1, -1).expand(*sizes[:1], -1)

        maskable_indices = self._shuffle_along_axis(maskable_indices, 1)

        indices_masked = maskable_indices[:, : self.n_masked_global]
        indices_unmasked = maskable_indices[:, self.n_masked_global :]

        return indices_masked, indices_unmasked

    def _gen_mask_both(self, sizes: tuple[int]) -> tuple[Tensor]:
        if self._mask_both_local:
            return self._gen_mask_local(sizes)
        else:
            return self._gen_mask_global(sizes)

    @staticmethod
    def reconstruct_batch(
        idx_masked: Tensor,
        idx_unmasked: Tensor,
        data_masked: Tensor,
        data_unmasked: Tensor,
    ) -> Tensor:
        """
        Reconstructs a tensor along the mask unit dimension. Batched version.

        Args:
            idx_masked: Tensor of shape `batch, mask unit sequence`.
            idx_unmasked: Tensor of shape `batch, mask unit sequence`.
            data_masked: Tensor of shape `batch, mask unit sequence, ...`.
                Should have same size along mask unit sequence dimension as
                idx_masked. Dimensions beyond the first two, marked here as ...
                will typically be `local_sequence, channel` or `channel, lat, lon`.
                  These dimensions should agree with data_unmasked.
            data_unmasked: Tensor of shape `batch, mask unit sequence, ...`.
                Should have same size along mask unit sequence dimension as
                idx_unmasked. Dimensions beyond the first two, marked here as
                ... will typically be `local_sequence, channel` or `channel,
                lat, lon`. These dimensions should agree with data_masked.
        Returns:
            Tensor of same shape as inputs data_masked and data_unmasked. I.e.
            `batch, mask unit sequence, ...`. Index for the total data composed
            of the masked and the unmasked part
        """
        dim: int = idx_masked.ndim

        idx_total = torch.argsort(torch.cat([idx_masked, idx_unmasked], dim=-1), dim=-1)
        idx_total = idx_total.view(*idx_total.shape, *[1] * (data_unmasked.ndim - dim))
        idx_total = idx_total.expand(*idx_total.shape[:dim], *data_unmasked.shape[dim:])

        data = torch.cat([data_masked, data_unmasked], dim=dim - 1)
        data = torch.gather(data, dim=dim - 1, index=idx_total)

        return data, idx_total

    def fourier_pos_encoding(self, x_static):
        """
        Args
            x_static: B x C x H x W. first two channels are lat, and lon respectively
        Returns
            Tensor of shape B x E x H x W where E is the embedding dimension.
        """

        # B x C x H x W -> B x 1 x H/P x W/P
        latitudes_patch = F.avg_pool2d(x_static[:, [0]], kernel_size=self.patch_size_px, stride=self.patch_size_px)
        longitudes_patch = F.avg_pool2d(x_static[:, [1]], kernel_size=self.patch_size_px, stride=self.patch_size_px)

        modes = torch.arange(self.embed_dim//4, device=x_static.device).view(1, -1, 1, 1) + 1.
        pos_encoding = torch.cat(
            (
                torch.sin(latitudes_patch*modes),
                torch.sin(longitudes_patch*modes),
                torch.cos(latitudes_patch*modes),
                torch.cos(longitudes_patch*modes),
            ),
            axis=1
        )

        return pos_encoding # B x E x H/P x W/P

    def time_encoding(self, input_time, lead_time):
        '''
        Args:
            input_time: Tensor of shape [batch].
            lead_time: Tensor of shape [batch].
        Returns:
            Tensor of shape [batch, embed_dim, 1, 1]
        '''
        input_time = self.input_time_embedding(input_time.view(-1, 1, 1, 1))
        lead_time = self.lead_time_embedding(lead_time.view(-1, 1, 1, 1))

        time_encoding = torch.cat(
            (
                torch.cos(input_time),
                torch.cos(lead_time),
                torch.sin(input_time),
                torch.sin(lead_time),
            ),
            axis=3
        )
        return time_encoding

    def to_patching(self, x: Tensor) -> Tensor:
        """Transform data from lat/lon space to two axis patching

        Args: ->
            x: Tesnor in lat/lon space (N, C, Nlat//P_0, Nlon//P_1)

        Returns:
            Tensor in patch space (N, G, L, C)
        """
        n_batch = x.shape[0]

        x = x.view(
            n_batch,
            self.embed_dim,
            self.global_shape_mu[0],
            self.local_shape_mu[0],
            self.global_shape_mu[1],
            self.local_shape_mu[1],
        )
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()

        s = x.shape
        return x.view(n_batch, s[1] * s[2], s[3] * s[4], -1)

    def from_patching(self, x: Tensor) -> Tensor:
        """Transform data from two axis patching to lat/lon space

        Args:
            x: Tensor in patch space with shape (N, G, L, C*P_0*P_1)

        Returns:
            Tensor in lat/lon space (N, C*P_0*P_1, Nlat//P_0, Nlon // P_1)
        """
        n_batch = x.shape[0]

        x = x.view(
            n_batch,
            self.global_shape_mu[0],
            self.global_shape_mu[1],
            self.local_shape_mu[0],
            self.local_shape_mu[1],
            -1,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()

        s = x.shape
        return x.view(n_batch, -1, s[2]*s[3], s[4]*s[5])

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: Dictionary containing the keys 'x', 'y', 'input_time',
                'lead_time' and 'static'. The associated torch tensors have the
                following shapes:
                x: Tensor of shape [batch, time, parameter, lat, lon]
                y: Tensor of shape [batch, parameter, lat, lon]
                static: Tensor of shape [batch, channel_static, lat, lon]
                climate: Optional tensor of shape [batch, parameter, lat, lon]
                input_time: Tensor of shape [batch]. Or none.
                lead_time: Tensor of shape [batch]. Or none.
        Returns:
            Tensor of shape [batch, parameter, lat, lon].
        """
        assert batch["x"].shape[2] == self.in_channels
        assert batch["x"].shape[3] == self.n_lats_px
        assert batch["x"].shape[4] == self.n_lons_px
        assert batch["y"].shape[1] == self.in_channels
        assert batch["y"].shape[2] == self.n_lats_px
        assert batch["y"].shape[3] == self.n_lons_px
        if self.positional_encoding == 'fourier':
            # the first two features (lat, lon) are encoded separately
            assert batch['static'].shape[1] - 2 == self.in_channels_static, "When setting self.positional_encoding to fourier, the number of static params change in the dataset. So, in the config, reduce num_static_channels (e.g., 4 instead of 7)."
        else:
            assert batch['static'].shape[1] == self.in_channels_static
        assert batch["static"].shape[2] == self.n_lats_px
        assert batch["static"].shape[3] == self.n_lons_px

        x_rescaled = (batch["x"] - self.input_scalers_mu) / (
            self.input_scalers_sigma + self.input_scalers_epsilon
        )
        batch_size = x_rescaled.shape[0]

        if self.positional_encoding == 'fourier':
            x_static_pos = self.fourier_pos_encoding(batch['static']) # B, embed_dim, lat / patch_size, lon / patch_size
            x_static = (batch['static'][:, 2:] - self.static_input_scalers_mu[:, 3:]) / ( # The first two channels in batch['static'] are used in positional encoding
                self.static_input_scalers_sigma[:, 3:] + self.static_input_scalers_epsilon # This translates to the first three channels in 'static_input_scalers_mu'
            )
        else:
            x_static = (batch["static"] - self.static_input_scalers_mu) / (
                self.static_input_scalers_sigma + self.static_input_scalers_epsilon
            )

        if self.residual == "temporal":
            # We create a residual of same shape as y
            index = torch.where(batch["lead_time"] > 0, batch["x"].shape[1] - 1, 0)
            index = index.view(-1, 1, 1, 1, 1)
            index = index.expand(batch_size, 1, *batch["x"].shape[2:])
            x_hat = torch.gather(batch["x"], dim=1, index=index)
            x_hat = x_hat.squeeze(1)
            assert (
                batch["y"].shape == x_hat.shape
            ), f'Shapes {batch["y"].shape} and {x_hat.shape} do not agree.'
        elif self.residual == "climate":
            climate_scaled = (
                batch["climate"] - self.input_scalers_mu.view(1, -1, 1, 1)
            ) / (
                self.input_scalers_sigma.view(1, -1, 1, 1) + self.input_scalers_epsilon
            )

        # [batch, time, parameter, lat, lon] -> [batch, time x parameter, lat, lon]
        x_rescaled = x_rescaled.flatten(1, 2)
        # Parameter dropout
        x_rescaled = self.parameter_dropout(x_rescaled)

        x_embedded = self.patch_embedding(x_rescaled)
        assert x_embedded.shape[1] == self.embed_dim

        if self.residual == "climate":
            static_embedded = self.patch_embedding_static(
                torch.cat((x_static, climate_scaled), dim=1)
            )
        else:
            static_embedded = self.patch_embedding_static(x_static)
        assert static_embedded.shape[1] == self.embed_dim

        if self.positional_encoding == 'fourier':
            static_embedded += x_static_pos

        x_embedded = self.to_patching(x_embedded)
        static_embedded = self.to_patching(static_embedded)

        time_encoding = self.time_encoding(batch['input_time'], batch['lead_time'])

        tokens = x_embedded + static_embedded + time_encoding

        # Now we generate masks based on masking_mode
        indices_masked, indices_unmasked = self.generate_mask(
            (batch_size, self._nglobal_mu)
        )
        indices_masked = indices_masked.to(device=tokens.device)
        indices_unmasked = indices_unmasked.to(device=tokens.device)
        maskdim: int = indices_masked.ndim

        # Unmasking
        unmask_view = (*indices_unmasked.shape, *[1] * (tokens.ndim - maskdim))
        unmasked = torch.gather(
            tokens,
            dim=maskdim - 1,
            index=indices_unmasked.view(*unmask_view).expand(
                *indices_unmasked.shape, *tokens.shape[maskdim:]
            ),
        )

        # Encoder
        x_encoded = self.encoder(unmasked)

        # Generate and position encode the mask tokens
        # (1, 1, 1, embed_dim) -> (batch, global_seq_masked, local seq, embed_dim)
        mask_view = (*indices_masked.shape, *[1] * (tokens.ndim - maskdim))
        masking = self.mask_token.repeat(*static_embedded.shape[:3], 1)
        masked = masking + static_embedded
        masked = torch.gather(
            masked,
            dim=maskdim - 1,
            index=indices_masked.view(*mask_view).expand(
                *indices_masked.shape, *tokens.shape[maskdim:]
            ),
        )

        recon, _ = self.reconstruct_batch(
            indices_masked, indices_unmasked, masked, x_encoded
        )

        x_decoded = self.decoder(recon)

        # Output: (batch, global sequence, local sequence, in_channels * patch_size[0] * patch_size[1])
        x_unembed = self.unembed(x_decoded)

        # Reshape to (batch, global_lat, global_lon, local_lat, local_lon, in_channels * patch_size[0] * patch_size[1])
        assert x_unembed.shape[0] == batch_size
        assert x_unembed.shape[1] == self.global_shape_mu[0] * self.global_shape_mu[1]
        assert x_unembed.shape[2] == self.local_shape_mu[0] * self.local_shape_mu[1]
        assert (
            x_unembed.shape[3]
            == self.in_channels * self.patch_size_px[0] * self.patch_size_px[1]
        )

        x_out = self.from_patching(x_unembed)

        # Pixel shuffle to (batch, in_channels, lat, lon)
        x_out = F.pixel_shuffle(x_out, self.patch_size_px[0])

        if self.residual == "temporal":
            x_out = self.output_scalers * x_out + x_hat
        elif self.residual == "climate":
            x_out = self.output_scalers * x_out + batch["climate"]
        elif self.residual == "none":
            x_out = self.output_scalers * x_out + self.input_scalers_mu.reshape(
                1, -1, 1, 1
            )

        return x_out
