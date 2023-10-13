import pandas as pd
from bng_latlon import OSGB36toWGS84
from config import map_im, out_dir_path, map_name
import os
try:
    from osgeo import gdal
except ModuleNotFoundError:
    pass

def convert_pixel_to_coord(bbox_x, bbox_y, image_path):
    tif = gdal.Open(image_path)
    gt = tif.GetGeoTransform()

    # gt contains top left corner coords and pixel sizes

    x_min = gt[0]
    x_size = gt[1]
    y_min = gt[3]
    y_size = gt[5]

    mx, my =  bbox_x,  bbox_y 
    lon_bng = mx * x_size + x_min
    lat_bng = my * y_size + y_min

    lat, lon  = OSGB36toWGS84(lon_bng, lat_bng) # convert to wgs84

    return lat, lon, lat_bng, lon_bng

def convert(out_dir_path, map_im, map_name):

    image_path = map_im

    df = pd.read_csv(os.path.join(out_dir_path, f'{map_name}_tree_coords_size.csv'))

    if df.empty:
        df.to_csv(f'{out_dir_path}/{map_name}_tree_coords_lat_lon.csv')
    
    else:
        df['lat'] = df.apply(lambda row : convert_pixel_to_coord(row['pixel_x_adjusted'], row['pixel_y_adjusted'], image_path)[0], axis = 1)
        df['lon'] = df.apply(lambda row : convert_pixel_to_coord(row['pixel_x_adjusted'], row['pixel_y_adjusted'], image_path)[1], axis = 1)
        df['lat_bng'] = df.apply(lambda row : convert_pixel_to_coord(row['pixel_x_adjusted'], row['pixel_y_adjusted'], image_path)[2], axis = 1)
        df['lon_bng'] = df.apply(lambda row : convert_pixel_to_coord(row['pixel_x_adjusted'], row['pixel_y_adjusted'], image_path)[3], axis = 1)
        df.to_csv(f'{out_dir_path}/{map_name}_tree_coords_lat_lon.csv', columns=['score','class','area','size','lat','lon','lat_bng','lon_bng'])


if __name__ == '__main__':
    convert(out_dir_path, map_im, map_name)