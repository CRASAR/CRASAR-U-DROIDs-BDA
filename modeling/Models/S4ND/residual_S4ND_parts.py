import math
import torch

from torch import nn


class ResidualBlockDoubleS4ND(nn.Module):
    def __init__(self,
                 N,
                 channels,
                 use_D=True,
                 lambda_scale=-0.1,
                 precision="half",
                 chunk_size=1,
                 band_limit_strategy="none",
                 learn_bandlimit=False,
                 cut_off_nyquist_proportion=0.25,
                 bandlimit_taper_width=0.1,
                 learn_frequency_importances=False,
                 frequency_importance_hidden_dim=32):
        super().__init__()

        self.double_s4nd = nn.Sequential(
                nn.GroupNorm(channels, channels),
                S4ND(channels=channels,
                     N=N,
                     use_D=use_D,
                     lambda_scale=lambda_scale,
                     precision=precision,
                     chunk_size=chunk_size,
                     band_limit_strategy=band_limit_strategy,
                     learn_bandlimit=learn_bandlimit,
                     cut_off_nyquist_proportion=cut_off_nyquist_proportion,
                     bandlimit_taper_width=bandlimit_taper_width,
                     learn_frequency_importances=learn_frequency_importances,
                     frequency_importance_hidden_dim=frequency_importance_hidden_dim),
                nn.ReLU(dim=1),
                nn.GroupNorm(channels, channels),
                S4ND(channels=channels,
                     N=N,
                     use_D=use_D,
                     lambda_scale=lambda_scale,
                     precision=precision,
                     chunk_size=chunk_size,
                     band_limit_strategy=band_limit_strategy,
                     learn_bandlimit=learn_bandlimit,
                     cut_off_nyquist_proportion=cut_off_nyquist_proportion,
                     bandlimit_taper_width=bandlimit_taper_width,
                     learn_frequency_importances=learn_frequency_importances,
                     frequency_importance_hidden_dim=frequency_importance_hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(x + self.double_s4nd(x))

class ResidualGroupS4ND(nn.Module):
    def __init__(self,
                 N,
                 in_channels,
                 out_channels,
                 num_blocks,
                 use_D=True,
                 lambda_scale=-0.1,
                 precision="half",
                 chunk_size=1,
                 band_limit_strategy="none",
                 learn_bandlimit=False,
                 cut_off_nyquist_proportion=0.25,
                 bandlimit_taper_width=0.1,
                 learn_frequency_importances=False,
                 frequency_importance_hidden_dim=32):
        super().__init__()

        self._blocks = []
        for i in range(0, num_blocks):
            self.blocks.append(ResidualBlockDoubleS4ND(
                    N=N,
                    channels=in_channels,
                    use_D=use_D,
                    lambda_scale=lambda_scale,
                    precision=precision,
                    chunk_size=chunk_size,
                    band_limit_strategy=band_limit_strategy,
                    learn_bandlimit=learn_bandlimit,
                    cut_off_nyquist_proportion=cut_off_nyquist_proportion,
                    bandlimit_taper_width=bandlimit_taper_width,
                    learn_frequency_importances=learn_frequency_importances,
                    frequency_importance_hidden_dim=frequency_importance_hidden_dim
                ))

        self._residual_group = nn.Sequential(
            self.blocks*, 
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._residual_group(x)
