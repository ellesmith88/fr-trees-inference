import pandas as pd
from config import out_dir_path

def get_size(scale, area, clss):
    if scale == '2500': # this is based on leeds 125642398 - tested on edi 2500 as well
        if clss == 'conifer':
            size = 'medium'
        else:# for broadleaves
            if area < 2000:
                size = 'small'
            if (area >= 2000) & (area < 3500):
                size = 'medium'
            if area >= 3500:
                size = 'large'

    elif scale == '500':
        if clss == 'conifer':
            if area < 10500:
                size = 'medium'
            else:
                size = 'large' # large about 15000 in edi sheets
        else:
            if area < 7500:
                size = 'small'
            if (area >= 7500) & (area < 20000):
                size = 'medium'
            if area >= 20000:
                size = 'large'


    elif scale == '1056':
        print('need to add size filters for this scale')

    return size

def apply_size_labels(scale, map_name):
    df = pd.read_csv(f'{out_dir_path}/{map_name}_tree_coords_nms.csv')

    df['area'] = (df['xmax']-df['xmin']) * (df['ymax']-df['ymin']) 
    df['size'] = df.apply(lambda row : get_size(scale, row['area'], row['class']), axis=1)

    df.to_csv(f'{out_dir_path}/{map_name}_tree_coords_size.csv')