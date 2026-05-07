from modeling.Models.S4ND.UNetS4ND import UNetS4ND
from modeling.Models.Model import TowerModule, MaskedTowerModule

class UNetS4NDTowerModule(TowerModule):
    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        return UNetS4ND(len(input_channel_map), len(output_label_map), hyperparameters=hyperparameters).cpu().to(self._device)

class MaskedUNetS4NDTowerModule(MaskedTowerModule):
    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        return UNetS4ND(len(input_channel_map), len(output_label_map), hyperparameters=hyperparameters).cpu().to(self._device)
