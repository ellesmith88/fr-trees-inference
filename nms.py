import pandas as pd
from config import out_dir_path, map_name, final_conifer_threshold, final_broadleaf_threshold

# code from
# https://github.com/vineeth2309/IOU/tree/main
# and
# https://medium.com/analytics-vidhya/non-max-suppression-nms-6623e6572536


def IOU(coords1, coords2):
    """ min coords are top left hand corner:
        coords1 = (xmin, ymin, xmax, ymax), and coords2 = (xmin2, ymin2, xmax2, ymax2) """

    x1, y1, x2, y2 = coords1
    
    x3, y3, x4, y4 = coords2

    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    
    width_inter = x_inter2 - x_inter1
    height_inter = y_inter2 - y_inter1
    
    # rejecting non overlapping boxes
    if height_inter <= 0 or width_inter <= 0:
        return
    
    area_inter = width_inter * height_inter
    
    area1 = (x2-x1) * (y2-y1)
    area2 = (x4-x3) * (y4-y3)
    area_union = (area1 + area2) - abs(area_inter)
    
    iou = area_inter / area_union
    return iou


def nms(boxes, conf_threshold=(final_conifer_threshold, final_broadleaf_threshold), iou_threshold=0.4):
    """
    The function performs nms on the list of boxes:
    boxes: [box1, box2, box3...]
    box1: [(xmin, ymin, xmax, ymax), Confidence, pixel x adjusted, pixel y adjusted, class]
    """
    final_conifer_threshold, final_broadleaf_threshold = conf_threshold
    bbox_list_thresholded = [] # List to contain the boxes after filtering by confidence
    bbox_list_new = [] # List to contain final boxes after nms 
    # Stage 1: (Sort boxes, and filter out boxes with low confidence)
    boxes_sorted = sorted(boxes, reverse=True, key = lambda x : x[1])	# Sort boxes according to confidence
    for box in boxes_sorted:
        if box[4] == 'conifer':
            if box[1] >= final_conifer_threshold: # Check if the box has a confidence greater than the threshold
                bbox_list_thresholded.append(box)	# Append the box to the list of thresholded boxes 
        elif box[4] == 'tree':
            if box[1] >= final_broadleaf_threshold: # Check if the box has a confidence greater than the threshold
                bbox_list_thresholded.append(box)	# Append the box to the list of thresholded boxes 
        else:
            pass
    #Stage 2: (Loop over all boxes, and remove boxes with high IOU)
    while len(bbox_list_thresholded) > 0:
        current_box = bbox_list_thresholded.pop(0) # Remove the box with highest confidence
        bbox_list_new.append(current_box) # Append it to the list of final boxes
        for box in bbox_list_thresholded:
            # if current_box[4] == box[4]: # Check if both boxes belong to the same class
                #import pdb;pdb.set_trace()
            iou = IOU(current_box[0], box[0]) # Calculate the IOU of the two boxes
            if iou is None: # they don't overlap
                continue

            if iou > iou_threshold: # Check if the iou is greater than the threshold defined
                bbox_list_thresholded.remove(box) # If there is significant overlap, then remove the box
    return bbox_list_new


def run_nms(map_name):
    df = pd.read_csv(f'{out_dir_path}/{map_name}_tree_coords.csv')

    if df.empty:
        df.to_csv(f'{out_dir_path}/{map_name}_tree_coords_nms.csv')

    else:

        tree_boxes = []
        # get out of data fram and pass through function
        for index, row in df.iterrows():
            tree_boxes.append([(row['xmin'],row['ymin'],row['xmax'],row['ymax']), row['score'], row['pixel_x_adjusted'], row['pixel_y_adjusted'], row['class']])
        nms_boxes = nms(tree_boxes)

        new_box_list = []

        
        for i in nms_boxes:
            new_box_list.append([i[0][0], i[0][1], i[0][2], i[0][3], i[1], i[2], i[3], i[4]])

        df_nms = pd.DataFrame(new_box_list, columns = ['xmin', 'ymin', 'xmax', 'ymax', 'score', 'pixel_x_adjusted', 'pixel_y_adjusted', 'class'])
        # save as df again
        df_nms.to_csv(f'{out_dir_path}/{map_name}_tree_coords_nms.csv')

if __name__ == '__main__':
    print(f'Running NMS for {map_name}')
    # run nms over results
    run_nms(map_name)
