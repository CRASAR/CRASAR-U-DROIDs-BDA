from collections import defaultdict

import torch
import shapely

from modeling.utils.sample_generator_utils import draw_buildings_on_mask, geometry_in_frame, draw_road_lines_on_mask
from modeling.Spatial import RoadLine, LabeledRoadLine
from modeling.DataMap import DefaultLabel2IdxMap

def divide_road_line_into_sub_segments(road_line, segment_length_pixels):
    segmented_road_line_geometry = road_line.getGeometry("pixels").segmentize(float(segment_length_pixels))
    subsegments = []
    xx, yy = segmented_road_line_geometry.coords.xy
    for i in range(0, len(xx)-1):
        subseg = [(xx[i], yy[i]), (xx[i+1], yy[i+1])]
        subsegments.append(RoadLine(
            identifier=None,
            geometry_source=road_line.getGeometrySource(),
            pixel_geom=shapely.LineString(subseg),
            epsg_4326_geom=None,
            adjusted=road_line.isAdjusted(),
            adjustment_subfield=road_line.getAdjustmentSubfield(),
            label=road_line.getLabel()
        ))

    return subsegments

def compute_masked_pixel_counts(preds, mask, label_to_idx_map):
    #Next we apply the mask to the predictions
    masked_data = None

    #Create a dictionary that will store the pixel counts
    result = {}

    #Stack the mask so it has the same dimension as the preds
    stacked_mask = mask.repeat(preds.shape[0], 1, 1)

    #Apply the mask to the data
    masked_data = stacked_mask * preds

    #Then, sum up along the class axes to get the total prediction mass for each class
    prediction_mass_by_class = masked_data.sum(dim=1).sum(dim=1)
    for label in label_to_idx_map.getAllLabels():
        result[label] = float(prediction_mass_by_class[label_to_idx_map.getIndex(label)])

    #return the results
    return result

#Function to map a preds tensor to labeled road lines via masking and counting/summing
def road_lines_to_labeled_road_line_segments(preds,
                                             road_lines,
                                             x_offset,
                                             y_offset,
                                             label_to_idx_map,
                                             segment_length_pixels=120,
                                             segment_buffer_width_pixels=40):
    resulting_segments = {}

    for road_line in road_lines:
        segments = divide_road_line_into_sub_segments(road_line, segment_length_pixels)
        resulting_segments[road_line] = {"parent":road_line, "segments":[]}
        for segment in segments:
            if geometry_in_frame(segment.getGeometry("pixels"), x_offset, y_offset, preds.shape[1], preds.shape[2]):
                #Generate a query mask where all the pixels that correspond to the segment are 1, and all the others are zero
                query = draw_road_lines_on_mask([segment],
                                                x_offset,
                                                y_offset,
                                                preds.shape[1],
                                                preds.shape[2],
                                                preds.shape[1],
                                                preds.shape[2],
                                                class_color_map=DefaultLabel2IdxMap(1),
                                                road_width_pixels=segment_buffer_width_pixels)

                #Reshape the query into the format of the preds
                mask = torch.tensor(query).reshape(1, preds.shape[1], preds.shape[2])

                #Count the number of pixels under the mask
                segment_pixel_counts = compute_masked_pixel_counts(preds, mask.to(preds.device), label_to_idx_map)

                #Get the label and confidence for this segment
                max_label, max_value = max(segment_pixel_counts.items(), key=lambda x:x[1])
                total_pixels = sum(segment_pixel_counts.values())

                #Store everything in the LabeledRoadLine object
                if total_pixels > 0:
                    resulting_segments[road_line]["segments"].append(LabeledRoadLine(identifier=None,
                                                                                     geometry_source=road_line.getGeometrySource(),
                                                                                     pixel_geom=segment.getGeometry("pixels"),
                                                                                     epsg_4326_geom=None,
                                                                                     adjusted=road_line.isAdjusted(),
                                                                                     adjustment_subfield=segment.getAdjustmentSubfield(),
                                                                                     label=max_label,
                                                                                     confidence=max_value/total_pixels,
                                                                                     parent_road_line_identifier=road_line.getId()))

    return resulting_segments

#Function to map a preds tensor to polygon IDs via masking and counting/summing
def buildings_to_pixel_counts(preds, buildings, x_offset, y_offset, label_to_idx_map):
    labels = {}

    #For every building we have to evaluate
    for building in buildings:

        #Get the x and y coordinate from the current polygon and offset it accordingly
        query = draw_buildings_on_mask([building],
                                       x_offset,
                                       y_offset,
                                       preds.shape[1],
                                       preds.shape[2],
                                       preds.shape[1],
                                       preds.shape[2],
                                       class_color_map=DefaultLabel2IdxMap(1))

        #Reshape the query into the format of the preds
        mask = torch.tensor(query).reshape(1, preds.shape[1], preds.shape[2])

        #Count the number of pixels under the mask
        labels[building.getId()] = compute_masked_pixel_counts(preds, mask.to(preds.device), label_to_idx_map)

    #Return the labels
    return labels

def combine_gathered_preds_dictionaries(outputs, accessor):
    result = defaultdict(list)
    for output in outputs:
        for k, v in accessor(output).items():
            result[k].extend(v)
    return result

def combine_gathered_labels(outputs, accessor):
    result = {}
    for output in outputs:
        for k, v in accessor(output).items():
            result[k] = v
    return result

def combine_gathered_loss(outputs, accessor):
    result = []
    for output in outputs:
        result.extend(accessor(output))
    return result
