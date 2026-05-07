from modeling.Models.Model import DecoderModule
from modeling.Models.Mask2Former.mask2former_head import Mask2FormerHead


class Mask2Former(DecoderModule):
    def load_decoder_model(self, hyperparameters, output_label_map):
        input_shape = {
            "1": (1024, None, None, 4),
            "2": (1024, None, None, 8),
            "3": (1024, None, None, 16),
            "4": (1024, None, None, 32),
        }
        return Mask2FormerHead(
            input_shape=input_shape,
            num_classes=len(output_label_map),
            transformer_in_feature=["1", "2", "3", "4"],
        )
