from rasterio.warp import calculate_default_transform, reproject, Resampling

import os
import argparse
import rasterio

def convert(input_folder):

    for root, dirs, files in os.walk(input_folder, topdown=True):
        for file in files:
            if file.endswith(".geo.tif"):
                input_geotif = os.path.join(root, file)
                print("Processing:" + str(input_geotif) + " .....")

                # Convert CRS to EPSG 32615
                output_folder = os.path.join(os.path.dirname(input_geotif), "converted")
                os.makedirs(output_folder, exist_ok=True)
                converted_geotif = os.path.join(output_folder, file)
                convert_to_epsg32615(input_geotif, converted_geotif)
                print("Saving converted ortho at " + str(converted_geotif))
                print("Done.")

def convert_to_epsg32615(input_geotif, output_geotif):
    with rasterio.open(input_geotif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:32615", src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'EPSG:32615',
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_geotif, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:32615",
                    resampling=Resampling.nearest
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='The input geotifs to be processed.')
    args = parser.parse_args()

    convert(args.input_folder)
