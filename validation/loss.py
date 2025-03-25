from typing import Optional

import numpy as np
import torch
import torch.nn as nn

class NormalizedMSELoss(nn.Module):
    """
    Normalized MSE Loss
    """

    def __init__(
        self,
        lats: list,
        feature_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            lats: List of latitudes, used to generate weighting (in degrees)
            feature_weights: Variance for each of the physical features
        """
        super().__init__()

        weights = np.array(np.cos(np.array(lats) * np.pi / 180.0))
        # Data is shaped (batch, feature, lat, lon)
        weights = weights.reshape(1, 1, -1, 1)
        assert not np.isnan(weights).any()
        self.weights = torch.tensor(weights, dtype=torch.float32)

        if feature_weights is not None:
            assert not torch.isnan(feature_weights).any()
            # Data is shaped (batch, feature, lat, lon)
            feature_weights = feature_weights.reshape(1, -1, 1, 1)
            self.feature_weights = feature_weights
        else:
            self.feature_weights = None

    def forward(self, y_hat: torch.Tensor, batch: dict[torch.Tensor]):
        """
        Calculate the loss

        Rescales both predictions and target, so assumes neither are already normalized
        Additionally weights by the cos(lat) of the set of features

        Args:
            pred: Prediction tensor
            target: Target tensor

        Returns:
            MSE loss on the variance-normalized values
        """
        y = batch['y']
        if not y_hat.shape == y.shape:
            raise ValueError(
                f'Unable to bring inputs y_hat and y with shapes {y_hat.shape} and {batch["y"].shape} into same shape.'
            )

        error = (y_hat - y) ** 2

        if self.weights.device != y_hat.device:
            self.weights = self.weights.to(y_hat.device)
        error = self.weights.expand_as(error) * error

        if self.feature_weights is not None:
            if self.feature_weights.device != y_hat.device:
                self.feature_weights = self.feature_weights.to(y_hat.device)
            error = self.feature_weights.expand_as(error) * error

        error = error.mean()
        assert not torch.isnan(error).any()

        return error