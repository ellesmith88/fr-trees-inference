from convert_pixels_to_coords import convert
from config import out_dir_path, get_map_name, city, scale, get_image_dir
from run_inference_pipeline import run
import argparse
from glob import glob
import os
import pandas as pd


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inf', action='store_true', help='Carries out inference on map images provided')
    parser.add_argument('-c', '--coords', action='store_true', help='Converts pixels to coords in csv files found in out_dir_path, then combines all into one csv')
    parser.add_argument('-mps', '--maps', help = 'path to directory where all map images are found')

    return parser.parse_args()

def loop():
    args = arg_parse()

    maps = glob(os.path.join(args.maps, '*.27.tif'))

    for m in maps:

        map_name = get_map_name(m)

        if args.inf:

            if os.path.isfile(os.path.join(out_dir_path, f'{map_name}_tree_coords_size.csv')):
                continue

            image_dir = get_image_dir(city, scale, map_name)
            run(image_dir, m, map_name)

        if args.coords:
            
            if os.path.isfile(os.path.join(out_dir_path, f'{map_name}_tree_coords_lat_lon.csv')):
                continue

            # convert pixel numbers to actual coordinates
            convert(out_dir_path, m, map_name)


    if args.coords:
        csvs = glob(os.path.join(out_dir_path, '*_tree_coords_lat_lon.csv'))
        df_concat = pd.concat([pd.read_csv(f) for f in csvs ], ignore_index=True)
        df_concat.to_csv(os.path.join(out_dir_path, f'combined_{city}_{scale}.csv'))



if __name__ == '__main__':
    loop()