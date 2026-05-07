import rasterio
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import cv2
import os
import argparse


def largest_black_component_mask(image):

    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(image)

    cv2.drawContours(mask, contours, -1, (255), cv2.FILLED)

    return mask


def plot_heatmaps(images, ortho_title, disaster, number_adj, area, save_path):
    xlabels = ["Imagery", "Distance", "Angle"]
    cmaps = ["", "viridis", "twilight"]
    # Create the figure and subplots

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))  # Adjust figsize as needed
    ims = []
    # Plot each image in a subplot
    for i, (ax, image, xlabel, cmap) in enumerate(zip(axes, images, xlabels, cmaps)):
        if not i:
            ax.imshow(image)  # Plot the image
        else:
            ims.append(ax.imshow(image, cmap=cmap))
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_xlabel(xlabel)  # Set x-axis label

    if len(images) > 1:
        # Colorbar for subplot 2
        fig.colorbar(ims[0], ax=axes[1], label="Pixels")
    if len(images) > 2:
        fig.colorbar(ims[1], ax=axes[2], label="Degrees")
    # Set the main title of the figure
    fig.suptitle(
        f"{ortho_title} | {disaster} | Number of Adjustments = {number_adj} | Area = {'{:.2f}'.format(area)} mi$^2$"
    )

    # Remove extra space between subplots (optional)
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0.2, wspace=0.2)

    try:
        plt.savefig(save_path + ortho_title + "_heatmap.png", bbox_inches="tight")
    except Exception as e:
        print(f"Save plot failed with error {e}")


def match_polygon_to_adjustment_ret_index(adjustments, polygon):
    min_dist = float("inf")
    best_line = None
    b_ind = 0

    for i, line in enumerate(adjustments):
        for vertex in polygon:
            x = vertex["x"]
            y = vertex["y"]
            dist = math.dist([x, y], line[0] // 8)
            if dist < min_dist:
                b_ind = i
                min_dist = dist
                best_line = line
    return b_ind


def ortho_heatmaps_pipeline(
    ortho_title, disaster, area, adjustments_path, input_geotif_path, output_path
):
    print("Calculating distances and angles")
    with open(adjustments_path) as json_file:
        adjs = np.array(json.load(json_file))
    angles = []
    distances = []
    for shift in adjs:
        distances.append(math.dist(shift[0], shift[1]))
        angles.append(
            (
                math.degrees(
                    math.atan2(
                        float(shift[0][1] - shift[1][1]),
                        float(shift[0][0] - shift[1][0]),
                    )
                )
                + 360
            )
            % 360
        )
    distances = np.array(distances)
    angles = np.array(angles)
    print("Done")
    print("Reading Ortho")

    # Read Image
    with rasterio.open(input_geotif_path) as src:
        image_data_full = src.read([1, 2, 3])
        image_data = src.read(1)
    print("Done")

    # Downscale by a factor of 8
    print("Downscaling and padding")
    image_data_full = np.transpose(image_data_full, (1, 2, 0))
    image_data = image_data[::2, ::2]
    image_data_full = image_data_full[::2, ::2]
    image_data = image_data[::2, ::2]
    image_data_full = image_data_full[::2, ::2]
    image_data = image_data[::2, ::2]
    image_data_full = image_data_full[::2, ::2]
    image_data = np.pad(image_data, pad_width=10, mode="constant", constant_values=0)
    image_data_full = np.pad(
        image_data_full,
        ((10, 10), (10, 10), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    mask = largest_black_component_mask(image_data)
    mask = mask.astype(np.uint8)
    print("Done")

    # Distance heatmap
    print("Calculating distance heatmap")
    heatmap = np.zeros((image_data.shape[0], image_data.shape[1]))
    for i in range(10, heatmap.shape[0], 10):
        for j in range(10, heatmap.shape[1], 10):
            heatmap[i - 10 : i, j - 10 : j] = distances[
                match_polygon_to_adjustment_ret_index(adjs, [{"x": j, "y": i}])
            ]

    heatmap_mask = np.zeros(heatmap.shape)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if mask[i, j] == 255:
                heatmap_mask[i, j] = heatmap[i, j]
    print("Done")

    print("Calculating angle heatmap")
    # Angle heatmap
    heatmap_ang = np.zeros((image_data.shape[0], image_data.shape[1]))
    for i in range(10, heatmap_ang.shape[0], 10):
        for j in range(10, heatmap_ang.shape[1], 10):
            heatmap_ang[i - 10 : i, j - 10 : j] = angles[
                match_polygon_to_adjustment_ret_index(adjs, [{"x": j, "y": i}])
            ]

    heatmap_ang_mask = np.full(heatmap.shape, 180)
    for i in range(heatmap_ang.shape[0]):
        for j in range(heatmap_ang.shape[1]):
            if mask[i, j] == 255:
                heatmap_ang_mask[i, j] = heatmap_ang[i, j]

    print("Done")
    print("Plotting heatmaps")
    plot_heatmaps(
        [image_data_full, heatmap_mask, heatmap_ang_mask],
        ortho_title,
        disaster,
        len(distances),
        area,
        output_path,
    )
    print("Done... deleting variables")
    del (
        image_data,
        image_data_full,
        heatmap,
        heatmap_ang,
        heatmap_ang_mask,
        heatmap_mask,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program generates heatmap plots for the adjustment with resepct to their lengths and angles"
    )
    parser.add_argument(
        "--adjustments_path", type=str, help="The path to the adjustments data"
    )
    parser.add_argument("--output_path", type=str, help="Path to the output folder")
    parser.add_argument("--input_geotif_path", type=str, help="Path to the geotif")
    parser.add_argument("--area", type=float, help="Area of the boundary", default=0.00)
    parser.add_argument(
        "--disaster", type=str, help="Disaster name", default="disaster-name"
    )
    args = parser.parse_args()

    ortho_title = os.path.split(args.input_geotif_path)[-1]
    print(f"Generating heatmap for {ortho_title}")
    ortho_heatmaps_pipeline(
        ortho_title,
        args.disaster,
        args.area,
        args.adjustments_path,
        args.input_geotif_path,
        args.output_path,
    )
