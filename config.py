try:
    import torch

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

except ModuleNotFoundError:
    pass

def get_map_name(map_im):
    map_name = map_im.split('\\')[-1].split('.')[0] 
    return map_name


# path to original rgb map image
map_im = '..\..\map_images\Edinburgh_1_500\\74415739.27.tif'

map_name = get_map_name(map_im)

# name of directory to store results - include image number 
out_dir_path = f'model/predictions/'

# classes: 0 index is reserved for background
CLASSES = [
    'background', 'tree', 'conifer'
]

# any detection having score below this will be discarded
detection_threshold = 0.6

model_path = 'model/grey_synthetic_200/best.pth'

scale = '500'
city= 'edi'

# path to directory that stores images to run model over
image_dir = f'..\split_ims\{city}\\1_{scale}\{map_name}\greyscale'

generate_imgs = False

# image slice parameters
slice_height = 512
slice_width = 512
# generally use 70 unless edi 1:500 - then use 150
y_overlap = 150
x_overlap = 150
