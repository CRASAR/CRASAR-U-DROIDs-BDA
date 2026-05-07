import torch
import torch.nn.functional as F

from torch import nn

class DynamicGroupLayerNorm(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, H, W = x.shape
        if H*W == 1:
            return F.layer_norm(x, [C, 1, 1])
        return super().forward(x)

NORMALIZATION_MAP = {
    "dynamicgrouplayernorm": [DynamicGroupLayerNorm, 2],
    "groupnorm": [nn.GroupNorm, 2],
    "batchnorm": [nn.BatchNorm2d, 1]
}

def getNormalization(name):
    return NORMALIZATION_MAP[name.lower().strip().replace("_", "")][0]
def getNormalizationArgCount(name):
    return NORMALIZATION_MAP[name.lower().strip().replace("_", "")][1]