from config import slice_height, slice_width, y_overlap, x_overlap, map_im, image_dir, map_name
import matplotlib.pyplot as plt
import pandas as pd
from skimage import color
import skimage
import os
import io


def calculate_slice_bboxes(
    im,
    slice_height,
    slice_width,
    y_overlap,
    x_overlap
) -> list[list[int]]:
    """
    Given the height and width of an image, calculates how to divide the image into
    overlapping slices according to the height and width provided. These slices are returned
    as bounding boxes in xyxy format.
    :return: a list of bounding boxes in xyxy format
    """
    try:
        image_height, image_width, _ = im.shape
    except ValueError:
        image_height, image_width = im.shape

    slice_bboxes = []
    y_max = y_min = 0
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


# def save_im(im, n):
#     #import pdb; pdb.set_trace()
#     plt.imsave(f'{image_dir}/block{n}.png', im, cmap=plt.cm.gray)


def split_image(im_grey, image_dir):

    img_list = []
    img = im_grey
    
    slices = calculate_slice_bboxes(img, slice_height, slice_width, y_overlap, x_overlap)

    # get tiles and create csv with tile corner pixel numers
    i = 0

    for s in slices:
        xmin, ymin, xmax, ymax = s
        im = img[ymin:ymax, xmin:xmax]

        plt.imsave(os.path.join(image_dir, f'block{i}.png'), im, cmap=plt.cm.gray, vmin=0, vmax=1)
        
        img_name = f'block{i}.png'
        img_dict = {'image_name': img_name, 'corner_x': xmin, 'corner_y': ymin}
        img_list.append(img_dict)

        i +=1

    print('number of blocks', len(slices))

    df = pd.DataFrame(img_list)
    df.to_csv(f'{image_dir}/img_coords.csv')

if __name__ == '__main__':
    im = skimage.io.imread(map_im)

    im_grey = color.rgb2gray(im)

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    print(f'Splitting image {map_name}')
    if os.path.isfile(os.path.join(image_dir, f'block0.png')):
        pass
    else:
        split_image(im_grey, image_dir)
