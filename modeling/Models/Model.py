import os

import torch
import torch.nn.functional as F
from torch import nn

import pytorch_lightning as L

from modeling.Models.Maskable import Maskable
from modeling.Models.ModelDatum import (
    ModelOutput,
    Y_HAT_SEGMENTATION_UNMASKED,
    Y_HAT_SEGMENTATION_MASKED,
    DO_SOFTMAX,
    MASK,
    CHANNEL_INPUT,
)


def _state_dict_backwards_compat(state_dict):
    for key in list(state_dict.keys()):
        if "frequency_importance_model" in key:
            del state_dict[key]
        else:
            if "model.model." in key:
                state_dict[key.replace("model.model.", "model.")] = state_dict.pop(key)
            else:
                state_dict[key.replace("model.", "")] = state_dict.pop(key)
    return state_dict


class SegmentationModelOutputModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def load(self, path, strict=True, assign=True):
        checkpoint = torch.load(path)
        try:
            # Assumption made that we only use load for inference
            checkpoint["state_dict"].pop("running_class_counts", None)
        except KeyError:
            pass

        try:
            del checkpoint["state_dict"]["running_class_counts"]
        except KeyError:
            pass

        try:
            self.load_state_dict(checkpoint["state_dict"], strict=strict, assign=assign)
        # pylint: disable-next=broad-exception-caught
        except Exception:
            compatible_state_dict = _state_dict_backwards_compat(checkpoint["state_dict"])
            self.load_state_dict(compatible_state_dict, strict=strict, assign=assign)

    def forward(self, model_input):
        result = ModelOutput()
        args = self.prepare_model_input_func(model_input)
        result.setField(Y_HAT_SEGMENTATION_UNMASKED, self.model(*args))
        scaled_result = self._interpolate_and_softmax(
            result, model_input[CHANNEL_INPUT].shape[-2:], model_input[DO_SOFTMAX]
        )
        return scaled_result

    def _interpolate_and_softmax(self, result, target_shape, do_softmax):
        result.setField(DO_SOFTMAX, do_softmax)

        # interpolating to make predications the expected size...
        preds = F.interpolate(
            result[Y_HAT_SEGMENTATION_UNMASKED],
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )

        if do_softmax:
            result.setField(Y_HAT_SEGMENTATION_UNMASKED, self.softmax(preds))
        else:
            result.setField(Y_HAT_SEGMENTATION_UNMASKED, preds)

        return result

    def prepare_model_input_func(self, model_input):
        return (model_input[CHANNEL_INPUT],)


# Towers are containers for models that we expect to run end to end as a single model and return ModelOutputs
# These are things like UNets which are not generally broken down into composite components except for the separate layers.
class Tower(SegmentationModelOutputModule):
    pass


# A Masked tower is an object that can be created that extends the existing tower logic and adds the masking logic to it also
class MaskedTower(Tower, Maskable):
    def __init__(
        self,
        model,
        n_cls,
        input_channel_mask_index=-1,
        output_channel_background_index=-1,
    ):
        Tower.__init__(self, model)
        Maskable.__init__(
            self, n_cls, input_channel_mask_index, output_channel_background_index
        )

    def forward(self, model_input):
        result = super().forward(model_input)
        if self.mask_output:
            result.setField(
                Y_HAT_SEGMENTATION_MASKED,
                self.mask(result[Y_HAT_SEGMENTATION_UNMASKED], model_input[MASK]),
            )
        return result


class TowerModule(L.LightningModule):
    def __init__(
        self, hyperparameters=None, input_channel_map=None, output_label_map=None
    ):
        super().__init__()
        self._model = self._load_tower_model(
            hyperparameters, input_channel_map, output_label_map
        )

    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        raise NotImplementedError(
            "_load_tower_model() has not yet been implemented by a subclass."
        )

    def get_model(self):
        return self._model

    def on_load_checkpoint(self, checkpoint):
        for key in list(checkpoint["state_dict"].keys()):
            checkpoint["state_dict"][key.replace("model.model.", "_model.model.")] = (
                checkpoint["state_dict"].pop(key)
            )


class MaskedTowerModule(TowerModule):
    def __init__(
        self, hyperparameters=None, input_channel_map=None, output_label_map=None
    ):
        super().__init__(hyperparameters, input_channel_map, output_label_map)

        # Initialize Tower
        model = self._load_tower_model(
            hyperparameters, input_channel_map, output_label_map
        )

        try:
            input_channel_mask_index = input_channel_map.getIdx("mask")
        except KeyError:
            print("Warning: There is no mask channel....Continuing without it...")
            input_channel_mask_index = -1

        # Initalize MaskedUperNet with backbone
        self._model = MaskedTower(
            model,
            len(output_label_map),
            input_channel_mask_index=input_channel_mask_index,
            output_channel_background_index=output_label_map.getBackgroundClassIdx(),
        )

    def _load_tower_model(self, hyperparameters, input_channel_map, output_label_map):
        raise NotImplementedError(
            "_load_tower_model() has not yet been implemented by a subclass."
        )


class EncoderModule:
    def __init__(self):
        pass

    def load_encoder_model(self, hyperparameters, output_label_map):
        raise NotImplementedError(
            "load_encoder_model() has not yet been implemented by a subclass."
        )

    def is_encoder(self):
        return True

    def is_decoder(self):
        return False


class DecoderModule:
    def __init__(self):
        pass

    def load_decoder_model(self, hyperparameters, output_label_map):
        raise NotImplementedError(
            "load_decoder_model() has not yet been implemented by a subclass."
        )

    def is_encoder(self):
        return False

    def is_decoder(self):
        return True


class EncodeDecode(SegmentationModelOutputModule):
    def __init__(self, encoder, decoder, prepare_model_input_func):
        super().__init__(None)
        self.softmax = nn.Softmax(dim=1)
        self.backbone = encoder
        self.decoder = decoder
        self.prepare_model_input_func = prepare_model_input_func

    def forward(self, model_input):
        args = self.prepare_model_input_func(model_input)
        intermediate = self.backbone(*args)
        y = self.decoder(intermediate)

        result = ModelOutput()
        result.setField(Y_HAT_SEGMENTATION_UNMASKED, y)
        return self._interpolate_and_softmax(
            result, model_input[CHANNEL_INPUT].shape[-2:], model_input[DO_SOFTMAX]
        )


class MaskedEncodeDecode(EncodeDecode, Maskable):
    def __init__(
        self,
        encoder,
        decoder,
        prepare_model_input_func,
        n_cls,
        input_channel_mask_index=-1,
        output_channel_background_index=-1,
    ):
        EncodeDecode.__init__(self, encoder, decoder, prepare_model_input_func)
        Maskable.__init__(
            self, n_cls, input_channel_mask_index, output_channel_background_index
        )

    def forward(self, model_input):
        result = super().forward(model_input)
        if self.mask_output:
            result.setField(
                Y_HAT_SEGMENTATION_MASKED,
                self.mask(result[Y_HAT_SEGMENTATION_UNMASKED], model_input[MASK]),
            )
        return result


class EncoderDecoderModule(L.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        hyperparameters=None,
        input_channel_map=None,
        output_label_map=None,
    ):
        super().__init__()

        # Initialize Encoder
        _backbone = encoder.load_encoder_model(hyperparameters, input_channel_map)

        # Initialize Decoder
        _decoder = decoder.load_decoder_model(hyperparameters, output_label_map)

        # Get the function to prepare inputs for the loaded encoder
        self._prep_model_input_func = encoder.prepare_model_input

        try:
            if hyperparameters["input"]["model_parameters"]["encoder_parameters"][
                "freeze_backbone"
            ]:
                print(
                    "NOTICE: Freezing Weights for backbone, this assumes pretrained backbone..\n"
                )
                for param in _backbone.parameters():
                    param.requires_grad = False
        except KeyError:
            pass

        # Initalize EncodeDecode with the models we have loaded
        self._model = EncodeDecode(_backbone, _decoder, self._prep_model_input_func)

    def get_model(self):
        return self._model

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            try:
                # Assumption made that we only use load for inference
                checkpoint["state_dict"].pop("running_class_counts", None)
            except KeyError:
                pass
            self.load_state_dict(checkpoint["model"])

    def on_load_checkpoint(self, checkpoint):
        for key in list(checkpoint["state_dict"].keys()):
            if key.startswith("model."):
                checkpoint["state_dict"][key.replace("model.", "_model.")] = checkpoint[
                    "state_dict"
                ].pop(key)


class MaskedEncoderDecoderModule(EncoderDecoderModule):
    def __init__(
        self,
        encoder,
        decoder,
        hyperparameters=None,
        input_channel_map=None,
        output_label_map=None,
    ):
        super().__init__(
            encoder, decoder, hyperparameters, input_channel_map, output_label_map
        )

        try:
            input_channel_mask_index = input_channel_map.getIdx("mask")
        except KeyError:
            print("Warning: There is no mask channel....Continuing without it...")
            input_channel_mask_index = -1

        # Initalize MaskedEncodeDecode with the models we have loaded
        self._model = MaskedEncodeDecode(
            self._model.backbone,
            self._model.decoder,
            self._prep_model_input_func,
            len(output_label_map),
            input_channel_mask_index=input_channel_mask_index,
            output_channel_background_index=output_label_map.getBackgroundClassIdx(),
        )
