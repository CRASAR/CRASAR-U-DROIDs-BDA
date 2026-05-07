import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

import rasterio
import numpy as np
import argparse

from dataset.visualizations.plot_ADJ_heatmaps import largest_black_component_mask

def count_pixels(image_name, input_geotif_path):
    print('Reading ortho...')
    with rasterio.open(os.path.join(input_geotif_path, image_name)) as src:
        image_data = src.read(1)
    mask = largest_black_component_mask(image_data)
    mask = mask.astype(np.uint8)
    print(f'Number of gigapixels in masked ortho are: {np.sum(mask>0, dtype=np.longlong)/1000000000}\n')

if __name__ == "__main__":
    
    # 
    #   Sample usage: 
    #   python count_pixels_area.py --image_name 20210901-Cocodrie-2.geo.tif  --input_geotif_path F:\HRAIL\imagery\UAS\
    # 

    parser = argparse.ArgumentParser(description='This program counts the number of pixels in a passed ortho and returns the count in gigapixels')
    parser.add_argument("--image_name", type=str,help="The name of the image that we want to count the pixels of", default='1002-Palm-Acers.4.geo.tif')
    parser.add_argument("--input_geotif_path", type=str, help="Path to the folder where all the geotifs are saved.")
    args = parser.parse_args()

    count_pixels(args.image_name, args.input_geotif_path)