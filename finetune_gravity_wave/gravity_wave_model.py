import torch
import torch.nn as nn
from PrithviWxC.model import PrithviWxC
from distributed import print0

torch.set_float32_matmul_precision("high")


class Encoder(nn.Module):
    """Encoder, consisting of a series of convolutional layers
    with batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of channels in the hidden layers.
    """

    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        # First encoding block
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Second encoding block
        self.encoder2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )

        # Third encoding block
        self.encoder3 = nn.Sequential(
            nn.Conv2d(
                hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Fourth encoding block
        self.encoder4 = nn.Sequential(
            nn.Conv2d(
                hidden_channels * 4, hidden_channels * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels * 8, hidden_channels * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Encoded tensors from each layer.
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        return enc1, enc2, enc3, enc4


class Decoder(nn.Module):
    """Decoder for UNet, consisting of convolutional layers to upsample and reconstruct the original input size.

    Args:
        hidden_channels (int): Number of hidden channels in the decoder layers.
        out_channels (int): Number of output channels.
    """

    def __init__(self, hidden_channels, out_channels):
        super(Decoder, self).__init__()

        # Fourth decoding block
        self.decoder4 = nn.Sequential(
            nn.Conv2d(
                hidden_channels * 16, hidden_channels * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels * 8, hidden_channels * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU(inplace=True),
        )

        # Third decoding block
        self.decoder3 = nn.Sequential(
            nn.Conv2d(
                hidden_channels * 12, hidden_channels * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Second decoding block
        self.decoder2 = nn.Sequential(
            nn.Conv2d(
                hidden_channels * 6, hidden_channels * 2, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )

        # First decoding block
        self.decoder1 = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Final output layer
        self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, enc1, enc2, enc3, enc4, bottleneck):
        """Forward pass for the decoder, concatenating the encoder outputs with the bottleneck.

        Args:
            enc1, enc2, enc3, enc4 (torch.Tensor): Encoder outputs.
            bottleneck (torch.Tensor): Bottleneck tensor from the transformer.

        Returns:
            torch.Tensor: Final output tensor.
        """
        dec4 = torch.cat((bottleneck, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = torch.cat((dec4, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = torch.cat((dec3, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = torch.cat((dec2, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        output = self.final_conv(dec1)
        return output


class UNetWithTransformer(nn.Module):
    """UNet model with a transformer-based bottleneck for climate data processing.

    Args:
        lr (float): Learning rate for the optimizer.
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        n_lats_px (int): Number of latitude pixels.
        n_lons_px (int): Number of longitude pixels.
        in_channels_static (int): Number of static input channels.
        mask_unit_size_px (list[int]): Size of masking units for the transformer.
        patch_size_px (list[int]): Size of patches for the transformer.
        device (str): Device to run the model on.
        ckpt_singular (str): Path to the checkpoint for pre-trained weights.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        in_channels: int = 488,
        hidden_channels: int = 160,
        out_channels: int = 366,
        n_lats_px: int = 64,
        n_lons_px: int = 128,
        in_channels_static: int = 3,
        mask_unit_size_px: list[int] = [8, 16],
        patch_size_px: list[int] = [2, 2],
        device="cpu",
        ckpt_singular=None,
    ):
        super().__init__()

        self.lr: float = lr
        self.patch_size_px: list[int] = patch_size_px
        self.out_channels: int = out_channels

        self.encoder = Encoder(in_channels, hidden_channels)
        self.decoder = Decoder(hidden_channels, out_channels)

        # Transformer model setup using PrithviWxC
        kwargs = {
            "in_channels": 1280,
            "input_size_time": 1,
            "n_lats_px": 64,
            "n_lons_px": 128,
            "patch_size_px": [2, 2],
            "mask_unit_size_px": [8, 16],
            "mask_ratio_inputs": 0.5,
            "embed_dim": 2560,
            "n_blocks_encoder": 12,
            "n_blocks_decoder": 2,
            "mlp_multiplier": 4,
            "n_heads": 16,
            "dropout": 0.0,
            "drop_path": 0.05,
            "parameter_dropout": 0.0,
            "residual": "none",
            "masking_mode": "both",
            "decoder_shifting": False,
            "positional_encoding": "absolute",
            "checkpoint_encoder": [3, 6, 9, 12, 15, 18, 21, 24],
            "checkpoint_decoder": [1, 3],
            "in_channels_static": 3,
            "input_scalers_mu": torch.tensor([0] * 1280),
            "input_scalers_sigma": torch.tensor([1] * 1280),
            "input_scalers_epsilon": 0,
            "static_input_scalers_mu": torch.tensor([0] * 3),
            "static_input_scalers_sigma": torch.tensor([1] * 3),
            "static_input_scalers_epsilon": 0,
            "output_scalers": torch.tensor([0] * 1280),
        }

        self.model = PrithviWxC(**kwargs)

        # Freeze transformer model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Load pre-trained weights if checkpoint is provided
        if ckpt_singular:

            print0(f"Starting to load model from {ckpt_singular}")
            state_dict = torch.load(
                f=ckpt_singular, map_location=device, weights_only=True
            )

            # Compare the keys in model and saved state_dict
            model_keys = set(self.model.state_dict().keys())
            saved_state_dict_keys = set(state_dict.keys())

            # Find keys that are in the model but not in the saved state_dict
            missing_in_saved = model_keys - saved_state_dict_keys
            # Find keys that are in the saved state_dict but not in the model
            missing_in_model = saved_state_dict_keys - model_keys
            # Find keys that are common between the model and the saved state_dict
            common_keys = model_keys & saved_state_dict_keys

            # Print the common keys
            if common_keys:
                print0(f"Keys loaded : {common_keys}")

            # Print the discrepancies
            if missing_in_saved:
                print0(f"Keys present in model but missing in saved state_dict: {missing_in_saved}")
            if missing_in_model:
                print0(f"Keys present in saved state_dict but missing in model: {missing_in_model}")

            # Load the state_dict with strict=False to allow partial loading
            self.model.load_state_dict(state_dict=state_dict, strict=False)
            print0('=>'*10, f"Model loaded from {ckpt_singular}...")
            print0("Loaded weights")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            batch (dict[str, torch.Tensor]): Dictionary containing input data, lead time, and static inputs.

        Returns:
            torch.Tensor: The final output of the model.
        """
        x = batch["x"]
        lead_time = batch["lead_time"]
        static = batch["static"]
        x = x.squeeze(1)

        # Encode input
        enc1, enc2, enc3, enc4 = self.encoder(x)

        # Reshape encoded data for the transformer
        batch_size, c, h, w = enc4.size()
        enc4_reshaped = enc4.unsqueeze(1)

        # Prepare input for transformer model
        batch_dict = {
            "x": enc4_reshaped,
            "y": enc4,
            "lead_time": lead_time,
            "static": static,
            "input_time": torch.zeros_like(lead_time),
        }

        # Transformer forward pass
        transformer_output = self.model(batch_dict)
        transformer_output_reshaped = transformer_output.view(batch_size, c, h, w)

        # Decode the transformer output
        output = self.decoder(enc1, enc2, enc3, enc4, transformer_output_reshaped)

        return output

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int = None
    ) -> torch.Tensor:
        """Perform a validation step.

        Args:
            batch (dict[str, torch.Tensor]): Input batch for validation.
            batch_idx (int, optional): Batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        y_hat: torch.Tensor = self(batch)

        # Compute loss
        loss: torch.Tensor = torch.nn.functional.mse_loss(
            input=y_hat, target=batch["target"]
        )
        return loss

    def get_model(self):
        """Return the encoder, decoder, and transformer model.

        Returns:
            Tuple[nn.Module, nn.Module, nn.Module]: The transformer, decoder, and encoder.
        """
        return self.model, self.decoder, self.encoder
