import numpy as np

import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from modeling.Models.TaskModuleBase import TaskModuleBase
from modeling.Models.OrthoInferenceWrapper import (
    fuse_bda_tiled_inference,
    joint_file_pred_key,
)
from modeling.utils.multiscale_metrics import (
    evaluate_multiscale_predictions,
    AUC_multiscale_metrics,
)
from modeling.formatters.plot_metrics_BDA import generate_confusion_matrix_plot
from modeling.utils.decoder_utils import (
    buildings_to_pixel_counts,
    combine_gathered_labels,
    combine_gathered_preds_dictionaries,
    combine_gathered_loss,
)


class TaskModuleBDA(TaskModuleBase):
    # pylint: disable-next=arguments-differ, unused-argument
    def predict_step(self, batch, batch_idx):
        y_hat_logits = self._compute_y_hat_inference(batch)
        y_hat_preds = F.softmax(y_hat_logits, dim=1)  # Obtain preds via softmax
        for buildings, ortho, gsd, y_hat_i in zip(
            batch.getBatchedBuildings(),
            batch.getBatchedOrthomosaic(),
            batch.getBatchedGSD(),
            y_hat_preds,
        ):
            if len(buildings) > 0:
                class_preds = buildings_to_pixel_counts(
                    y_hat_i, buildings, 0, 0, label_to_idx_map=self.output_label_map
                )
                for b in buildings:
                    self.predict_step_outputs[
                        joint_file_pred_key(ortho.get_name(), b.getId(), gsd[0], gsd[1])
                    ].append({"class_preds": class_preds[b.getId()]})

    def on_predict_epoch_end(self):
        local_dict = {"preds": self.predict_step_outputs}
        world_size = self.trainer.world_size
        gathered = [None for _ in range(world_size)]
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(gathered, local_dict)
            global_dict = {
                "preds": combine_gathered_preds_dictionaries(
                    gathered, lambda x: x["preds"]
                )
            }
        else:
            global_dict = local_dict

        self.predicted_labels = fuse_bda_tiled_inference(global_dict["preds"])

    # pylint: disable-next=arguments-differ, unused-argument
    def validation_step(self, batch, batch_idx):
        y_hat_logits = self._compute_y_hat_inference(batch)
        y_hat_preds = F.softmax(y_hat_logits, dim=1)  # Obtain preds via softmax
        for buildings, ortho, gsd, y_hat_i in zip(
            batch.getBatchedBuildings(),
            batch.getBatchedOrthomosaic(),
            batch.getBatchedGSD(),
            y_hat_preds,
        ):
            if len(buildings) > 0:
                class_preds = buildings_to_pixel_counts(
                    y_hat_i, buildings, 0, 0, label_to_idx_map=self.output_label_map
                )
                for b in buildings:
                    joint_id = joint_file_pred_key(
                        ortho.get_name(), b.getId(), gsd[0], gsd[1]
                    )
                    self.validation_step_outputs[joint_id].append(
                        {"class_preds": class_preds[b.getId()]}
                    )
                    self.validation_step_labels[joint_id] = b.getLabel()
                    for c in self.output_label_map.getAllLabels():
                        self._step_metadata.scalars["val/Predicted_Pixel_Counts"][
                            c
                        ] += float(class_preds[b.getId()][c])
                self._step_metadata.normalizations["val/Predicted_Pixel_Counts"] += len(
                    buildings
                )

        # Compute the crtierion loss
        criterion_loss = (
            self.criterion(y_hat_logits, batch.getBatchedLabels())
            * self.criterion_scale_factor
        )
        self.validation_loss.append(criterion_loss.detach().cpu().tolist())

    def on_validation_epoch_end(self):
        local_dict = {
            "labels": self.validation_step_labels,
            "preds": self.validation_step_outputs,
            "loss": self.validation_loss,
        }

        world_size = self.trainer.world_size
        gathered = [None for _ in range(world_size)]
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(gathered, local_dict)
            global_dict = {
                "labels": combine_gathered_labels(gathered, lambda x: x["labels"]),
                "preds": combine_gathered_preds_dictionaries(
                    gathered, lambda x: x["preds"]
                ),
                "loss": combine_gathered_loss(gathered, lambda x: x["loss"]),
            }
        else:
            global_dict = local_dict

        fused_preds = fuse_bda_tiled_inference(global_dict["preds"])
        for c in self.output_label_map.getAllLabels():
            self._step_metadata.scalars["val/Predicted_Class_Counts"][c] = 0
        for pred in fused_preds.values():
            self._step_metadata.scalars["val/Predicted_Class_Counts"][
                pred["label"]
            ] += 1

        actual_labels = []
        preds_labels = []
        for building_id, actual_label in global_dict["labels"].items():
            actual_labels.append(actual_label)
            preds_labels.append(fused_preds[building_id]["label"])

        macro_f1 = f1_score(actual_labels, preds_labels, average="macro")
        micro_f1 = f1_score(actual_labels, preds_labels, average="micro")
        macro_precision = precision_score(actual_labels, preds_labels, average="macro")
        micro_precision = precision_score(actual_labels, preds_labels, average="micro")
        macro_recall = recall_score(actual_labels, preds_labels, average="macro")
        micro_recall = recall_score(actual_labels, preds_labels, average="micro")

        bda_confusion_matrix = confusion_matrix(
            y_true=actual_labels,
            y_pred=preds_labels,
            labels=list(self.output_label_map.getAllLabels()),
        )
        matrix_data = {
            "Confusion_Matrix": {
                "matrix": bda_confusion_matrix.tolist(),
                "class_labels": self.output_label_map.getAllLabels(),
            }
        }

        np_image = generate_confusion_matrix_plot(
            [
                {
                    "metrics": matrix_data,
                    "samples": {"total": len(global_dict["labels"].keys())},
                    "step": self.global_step,
                }
            ],
            [self.getName()],
            return_np=True,
        )

        gsd_bucketed_macro_f1 = evaluate_multiscale_predictions(
            fused_preds,
            global_dict["labels"],
            lambda actuals, preds: f1_score(actuals, preds, average="macro"),
        )
        gsd_bucketed_micro_f1 = evaluate_multiscale_predictions(
            fused_preds,
            global_dict["labels"],
            lambda actuals, preds: f1_score(actuals, preds, average="micro"),
        )
        gsd_bucketed_macro_precision = evaluate_multiscale_predictions(
            fused_preds,
            global_dict["labels"],
            lambda actuals, preds: precision_score(actuals, preds, average="macro"),
        )
        gsd_bucketed_micro_precision = evaluate_multiscale_predictions(
            fused_preds,
            global_dict["labels"],
            lambda actuals, preds: precision_score(actuals, preds, average="micro"),
        )
        gsd_bucketed_macro_recall = evaluate_multiscale_predictions(
            fused_preds,
            global_dict["labels"],
            lambda actuals, preds: recall_score(actuals, preds, average="macro"),
        )
        gsd_bucketed_micro_recall = evaluate_multiscale_predictions(
            fused_preds,
            global_dict["labels"],
            lambda actuals, preds: recall_score(actuals, preds, average="micro"),
        )

        auc_macro_f1 = AUC_multiscale_metrics(
            gsd_bucketed_macro_f1, log_space_area=True, normalize=True
        )
        auc_micro_f1 = AUC_multiscale_metrics(
            gsd_bucketed_micro_f1, log_space_area=True, normalize=True
        )
        auc_macro_precision = AUC_multiscale_metrics(
            gsd_bucketed_macro_precision, log_space_area=True, normalize=True
        )
        auc_micro_precision = AUC_multiscale_metrics(
            gsd_bucketed_micro_precision, log_space_area=True, normalize=True
        )
        auc_macro_recall = AUC_multiscale_metrics(
            gsd_bucketed_macro_recall, log_space_area=True, normalize=True
        )
        auc_micro_recall = AUC_multiscale_metrics(
            gsd_bucketed_micro_recall, log_space_area=True, normalize=True
        )

        self.log("val_macro_f1", macro_f1)
        self.log("auc_macro_f1", auc_macro_f1)

        if self.trainer.is_global_zero:
            logger = self.get_tb_logger()
            logger.add_image(
                "val_global/ConfusionMatrix",
                np_image,
                self.global_step,
                dataformats="HWC",
            )
            logger.add_scalar(
                "val_global/criterion_loss",
                np.mean(global_dict["loss"]),
                self.global_step,
            )
            logger.add_scalar("val_global/macro_f1", macro_f1, self.global_step)
            logger.add_scalar("val_global/micro_f1", micro_f1, self.global_step)
            logger.add_scalar(
                "val_global/macro_precision", macro_precision, self.global_step
            )
            logger.add_scalar(
                "val_global/micro_precision", micro_precision, self.global_step
            )
            logger.add_scalar("val_global/macro_recall", macro_recall, self.global_step)
            logger.add_scalar("val_global/micro_recall", micro_recall, self.global_step)

            logger.add_scalar(
                "val_multiscale/auc_macro_f1", auc_macro_f1, self.global_step
            )
            logger.add_scalar(
                "val_multiscale/auc_micro_f1", auc_micro_f1, self.global_step
            )
            logger.add_scalar(
                "val_multiscale/auc_macro_precision",
                auc_macro_precision,
                self.global_step,
            )
            logger.add_scalar(
                "val_multiscale/auc_micro_precision",
                auc_micro_precision,
                self.global_step,
            )
            logger.add_scalar(
                "val_multiscale/auc_macro_recall", auc_macro_recall, self.global_step
            )
            logger.add_scalar(
                "val_multiscale/auc_micro_recall", auc_micro_recall, self.global_step
            )

            logger.add_scalars(
                "val_multiscale/macro_f1",
                {str(k): v for k, v in gsd_bucketed_macro_f1.items()},
                self.global_step,
            )
            logger.add_scalars(
                "val_multiscale/micro_f1",
                {str(k): v for k, v in gsd_bucketed_micro_f1.items()},
                self.global_step,
            )
            logger.add_scalars(
                "val_multiscale/macro_precision",
                {str(k): v for k, v in gsd_bucketed_macro_precision.items()},
                self.global_step,
            )
            logger.add_scalars(
                "val_multiscale/micro_precision",
                {str(k): v for k, v in gsd_bucketed_micro_precision.items()},
                self.global_step,
            )
            logger.add_scalars(
                "val_multiscale/macro_recall",
                {str(k): v for k, v in gsd_bucketed_macro_recall.items()},
                self.global_step,
            )
            logger.add_scalars(
                "val_multiscale/micro_recall",
                {str(k): v for k, v in gsd_bucketed_micro_recall.items()},
                self.global_step,
            )

            # Assemble the alert that will be sent
            alert_messages = [
                f"\nValidation #{self.current_epoch} on step {self.global_step}",
                f"\tval_global/macro_f1: {macro_f1:0.5f}",
                f"\tval_multiscale/auc_macro_f1: {auc_macro_f1:0.5f}",
                "\tval_multiscale/macro_f1:",
            ]
            for k, v in gsd_bucketed_macro_f1.items():
                alert_messages.append(f"\t\tmacro_f1: {v:0.5f} | gsd: {k:0.5f}")

            self.sendAlert("\n".join(alert_messages))
