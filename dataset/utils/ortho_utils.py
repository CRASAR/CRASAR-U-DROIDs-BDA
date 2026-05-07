import rasterio
from rasterio.features import dataset_features
from rasterio.warp import transform_bounds as transform_bounds_rio
from rasterio.coords import BoundingBox
from rasterio.windows import Window
from rasterio.enums import Resampling

def crop_to_raw(input_data, x, y, width, height):
	# Create a Window and calculate the transform from the source dataset    
	width_adj = min(width, (x+width)-input_data.width)
	height_adj = min(height, (x+height)-input_data.height)

	window = Window(x, y, width_adj, height_adj)
	return input_data.read(window=window)

def crop_to_file(input_data, output_path, x, y, width, height):
	# Create a Window and calculate the transform from the source dataset

	width_adj = min(width, (x+width)-input_data.width)
	width_adj = width if width_adj < 0 else width-width_adj
	height_adj = min(height, (y+height)-input_data.height)
	height_adj = height if height_adj < 0 else height-height_adj
	
	window = Window(x, y, width_adj, height_adj)
	transform = input_data.window_transform(window)

	# Create a new cropped raster to write to
	profile = input_data.profile
	profile.update({
		'height': height_adj,
		'width': width_adj,
		'transform': transform})

	with rasterio.open(output_path, 'w', **profile) as dst:
		# Read the data from the window and write it to the output raster
		dst.write(input_data.read(window=window))


def transform_bounds(input_data, target_crs):

	left, bottom, right, top = transform_bounds_rio(input_data.crs, {"init":str(target_crs)}, *input_data.bounds)
	return BoundingBox(left, bottom, right, top)

def compute_ortho_polygon(input_data):
	shapes = list(dataset_features(input_data, bidx=1, as_mask=True, geographic=True, band=True))
	return shapes

