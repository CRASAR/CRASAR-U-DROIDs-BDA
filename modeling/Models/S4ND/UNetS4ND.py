from torch import nn

from torch.utils.checkpoint import checkpoint

from modeling.Models.MaskedUNet.unet_parts import AttnUp, Up, OutConv
from modeling.Models.S4ND.S4ND_parts import DoubleS4ND, S4NDDown

class UNetS4ND(nn.Module):
    def __init__(self, n_channels, n_classes, hyperparameters):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self._use_checkpoints = hyperparameters["model_parameters"]["checkpointing"]

        self.inc = DoubleS4ND(N=hyperparameters["model_parameters"]["layers"]["inc"]["N"],
                              in_channels=n_channels,
                              out_channels=hyperparameters["model_parameters"]["layers"]["inc"]["out_channels"],
                              activation=hyperparameters["model_parameters"]["layers"]["inc"]["activation"],
                              normalization=hyperparameters["model_parameters"]["layers"]["inc"]["normalization"])

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        last_up_out_channels = None
        for layer in hyperparameters["model_parameters"]["layers"].keys():
            if "down_" in layer:
                l = S4NDDown(**hyperparameters["model_parameters"]["layers"][layer])
                self.down_layers.append(l)
            if "up_" in layer:
                up_module = AttnUp if hyperparameters["model_parameters"]["layers"][layer]["attention"] else Up
                factor = 2 if hyperparameters["model_parameters"]["layers"][layer]["bilinear"] else 1
                l = up_module(in_channels=hyperparameters["model_parameters"]["layers"][layer]["in_channels"],
                              out_channels=hyperparameters["model_parameters"]["layers"][layer]["out_channels"] // factor,
                              bilinear=hyperparameters["model_parameters"]["layers"][layer]["bilinear"],
                              activation=hyperparameters["model_parameters"]["layers"][layer]["activation"],
                              normalization=hyperparameters["model_parameters"]["layers"][layer]["normalization"])

                last_up_out_channels = hyperparameters["model_parameters"]["layers"][layer]["out_channels"] // factor
                self.up_layers.append(l)

        #Because this is a unet we assert that there are at least as many down layers as up layers
        assert(len(self.down_layers) == len(self.up_layers))

        self.outc = OutConv(last_up_out_channels, n_classes)

    def _compute_down_layers(self, x):
        down_outs = [x]
        for l in self.down_layers:
            if self._use_checkpoints:
                val = checkpoint(l, down_outs[-1], use_reentrant=False)
            else:
                val = l(down_outs[-1])
            down_outs.append(val)
        return down_outs

    def _compute_up_layers(self, down_outs):
        intermediate = down_outs[-1]
        for l, d in zip(self.up_layers, down_outs[-2::-1]):
            if self._use_checkpoints:
                intermediate = checkpoint(l, intermediate, d, use_reentrant=False)
            else:
                intermediate = l(intermediate, d)
        return intermediate

    def forward(self, x):
        x_inc = self.inc(x)
        down_outs = self._compute_down_layers(x_inc)
        up_convolved_rep = self._compute_up_layers(down_outs)
        return self.outc(up_convolved_rep)
