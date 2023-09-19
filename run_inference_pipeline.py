from config import map_im, out_dir_path, generate_imgs, scale, map_name, image_dir
from skimage import color
from skimage import io
from split_image import split_image
from inference import run_inference
from nms import run_nms
from get_size_label import apply_size_labels
import os

# convert original image to greyscale
def convert_to_grey(im):
    im_grey = color.rgb2gray(im)

    return im_grey


def run(image_dir, map_im=map_im, map_name=map_name):
    # get original image and convert to greyscale
    im = io.imread(map_im)
    im_grey = convert_to_grey(im)

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # split image up into many patches
    print(f'Splitting image {map_name}')
    if os.path.isfile(os.path.join(image_dir, f'block0.png')):
        pass
    else:
        split_image(im_grey, image_dir)

    print(f'Runnning inference for {map_name}')
    # run model over these patches for prediction - output is saved in results directory
    run_inference(scale, out_dir_path, map_name, generate_imgs, image_dir)

    print(f'Running NMS for {map_name}')
    # run nms over results
    run_nms(map_name)

    print(f'Applying size labels for {map_name}')
    # run size classifier over results
    apply_size_labels(scale, map_name)


    print(f'Complete for {map_name}')

if __name__ == '__main__':
    run(image_dir, map_im, map_name)