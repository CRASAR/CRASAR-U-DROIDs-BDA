from modeling.Models.MaskedUNet.UNet import UNet
from modeling.Models.Model import TowerModule, MaskedTowerModule

class UNetTowerModule(TowerModule):
    def _load_tower_model(self, hyperparameters, _, output_label_map):
        return UNet(hyperparameters["input"]["in_channels"], len(output_label_map), hyperparameters=hyperparameters).cpu().to(self._device)

class MaskedUNetTowerModule(MaskedTowerModule):
    def _load_tower_model(self, hyperparameters, _, output_label_map):
        return UNet(hyperparameters["input"]["in_channels"], len(output_label_map), hyperparameters=hyperparameters).cpu().to(self._device)
