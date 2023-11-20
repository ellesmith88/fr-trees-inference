try:
    import torch

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

except ModuleNotFoundError:
    pass

def get_map_name(map_im):
    map_name = map_im.split('\\')[-1].split('.')[0] 
    return map_name


# path to original rgb map image
#map_im = '..\..\map_images\Edinburgh_1_500\\74417532.27.tif'
map_im = '..\..\map_images\\Leeds_1_500\\229947114.27.tif'

map_name = get_map_name(map_im)

# name of directory to store results - include image number 
out_dir_path = f'model/leeds_1889_predictions_high_conf/'

# classes: 0 index is reserved for background
CLASSES = [
    'background', 'tree', 'conifer'
]

# any detection having score below this will be discarded
initial_detection_threshold = 0.7
final_conifer_threshold = 0.849 
final_broadleaf_threshold = 0.972

#leeds vals
#con = 0.849 
#broadleaf = 0.972

#edi vals
#con = 0.975
#broadleaf = 0.984 

model_path = 'model/extra/best.pth'

scale = '500'
city= 'leeds'

def get_image_dir(city, scale, map_name):
    # path to directory that stores images to run model over
    image_dir = f'..\split_ims\{city}\\1_{scale}\{map_name}\greyscale'
    return image_dir

image_dir = get_image_dir(city, scale, map_name)

generate_imgs = False

# image slice parameters
slice_height = 512
slice_width = 512
# generally use 70 unless 1:500 - then use 150 for edi, 165 for leeds
y_overlap = 165
x_overlap = 165
