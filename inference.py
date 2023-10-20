import numpy as np
import cv2
from glob import glob
import os
from model import create_model
import pandas as pd
from config import image_dir, CLASSES, detection_threshold, model_path, out_dir_path, scale, map_name, generate_imgs


def set_up_computation_device():
    try:
        import torch
    except ModuleNotFoundError:
        pass

    # set the computation device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model and the trained weights
    model = create_model(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device
    ))

    model.eval()

    return model

def adjust_pixel_coords(bbox_x, bbox_y, image_name, path):
    df = pd.read_csv(os.path.join(path,'img_coords.csv'))

    corner_y = int(df[df['image_name']==f'{image_name}.png']['corner_y'])
    corner_x = int(df[df['image_name']==f'{image_name}.png']['corner_x'])

    x_new = bbox_x + int(corner_x)
    y_new = bbox_y + int(corner_y)

    return x_new, y_new, corner_x, corner_y


def run_inference(scale, out_dir_name, map_name, generate_imgs, image_dir=image_dir):
    '''
    scale (str): scale of the map sheet (1056, 500, 2500)
    out_dir_name (str): name of directory to output results in
    generate_imgs (bool): whether images with bounding boxes should be created or not
    '''

    model = set_up_computation_device()

    try:
        import torch
    except ModuleNotFoundError:
        pass

    coord_list = []
 
    images = glob(f"{image_dir}/block*.png")
    print(f"Images to evaluate: {len(images)}")

    for i in range(len(images)):

        # get the image file name for saving output later on
        image_name = images[i].split('\\')[-1].split('.')[0]

        #print(image_name)

        image = cv2.imread(images[i])
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(float)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)
        
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
        
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
    
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            labels = outputs[0]['labels'].data.numpy()[scores >= detection_threshold]
            scores = scores[scores >= detection_threshold]

            draw_boxes = boxes.copy()
            
            labels_list = labels.tolist()
            # get all the predicited class names - need to filter this according to scores
            pred_classes = [CLASSES[i] for i in labels_list]
            
            # iterate through detected boxes
            for j, box in enumerate(draw_boxes):

                # add to pandas dataframe - middle point of bounding box
                # image name, top left corner pixel x and y, pixel x, pixel y, pixel x and y adjusted, lat, lon, 
                if scale == '500' or '1056':
                    x_new, y_new, corner_x, corner_y = adjust_pixel_coords((box[0]+box[2])/2, (box[1]+box[3])/2, image_name, image_dir) #- mid point of box
                    
                else:
                    x_new, y_new, corner_x, corner_y = adjust_pixel_coords((box[0]+box[2])/2, box[3], image_name, image_dir) # middle of base of box

                xmin, ymin = adjust_pixel_coords(box[0], box[1], image_name, image_dir)[0:2]
                xmax, ymax = adjust_pixel_coords(box[2], box[3], image_name, image_dir)[0:2]

                coord_dict = {'image_name': image_name, 'corner_x': corner_x, 'corner_y': corner_y, 'pixel_x':(box[0]+box[2])/2, 'pixel_y':box[3], 'pixel_x_adjusted':x_new, 'pixel_y_adjusted':y_new, 'score':scores[j], 'width':box[2]-box[0], 'height':box[3]-box[1], 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'class': pred_classes[j]}
                coord_list.append(coord_dict)

                # draw the bounding boxes and write the class name on top of it
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)
                

                cv2.putText(orig_image, f'{pred_classes[j]}', 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                            2, lineType=cv2.LINE_AA)
                
                

                if generate_imgs is True:
                    if not os.path.exists(os.path.join(out_dir_path, map_name)):
                    # Create a new directory because it does not exist
                        path = os.path.join(out_dir_path, map_name)
                        os.makedirs(path)

                        cv2.imwrite(f"{out_dir_path}/{map_name}/{image_name}.png", orig_image)

   
        #print('-'*50)


        df = pd.DataFrame(coord_list)
        df.to_csv(f'{out_dir_name}/{map_name}_tree_coords.csv')

if __name__ == '__main__':
    print(f'Runnning inference for {map_name}')
    # run model over these patches for prediction - output is saved in results directory
    run_inference(scale, out_dir_path, map_name, generate_imgs, image_dir)

