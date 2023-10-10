# fr-trees-inference

This code is based on the tutorial at https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/ by Sovit Ranjan Rath

This code is used to run inference over historic OS map sheets to identify tree symbols. The model weights must be provided.

There are 2 methods of use - run over one map sheet or run as a loop over several maps sheets. The maps sheets must be GeoTiffs to extract the latitude and longitude information. It is assumed that maps are named as the examples: ``74417532.27.tif`` or ``74417532.1.tif`` where ``74417532`` is the map name.

How to use:

**Before starting:** create a new conda environment and install the required packages, in environment.yml
It may be necessary to have ``gdal`` installed in separate environment as there may be conflicts with ``pytorch``. The code is setup to allow the necessary code to run without ``pytorch`` and vice-versa.

1. ``config.py`` contains all the settings required and should be adjusted before running any code.
- ``map_im`` is the path to the original map image you want to run inference over, if only using the code for one sheet.
- ``out_dir_path`` is the name of the directory to store results
- ``CLASSES`` is a list of classes the model should detect (same as the model was trained with)
- ``detection threshold`` determines the score below which detections will be discarded.
- ``model_path`` is the path to te model weights
- ``scale`` is the scale of the maps that trees are being identified on. '500' is 1:500, '2500' is 1:2500 etc.
- ``city`` is the city of the maps being used
- ``generate_imgs`` should be True of False. If True, images will be generated showing the bounding boxes around idnetfied objects.
- ``slice_height`` is the height of 'patches' when a whole image is split up (in pixels)
- ``slice_width`` is the width of 'patches' when a whole image is split up (in pixels)
- ``y_overlap`` is the vertical overlap (in pixels) for pathces when a whole image is split up
- ``x_overlap`` is the horizontal overlap (in pixels) for pathces when a whole image is split up

2. If running through several sheets in a loop, use the ``loop.py`` script.
  
   ``loop.py`` must be run with the ``-mps`` argument where the path to the directory storing all of the images must be provided. There are then two options:  

   Firstly, run with arguemnt ``-i`` to carry out the inference. For each image, the image is converted to greyscale and split, inference is carried out, NMS (non-maximum suppression) is applied to the results and size labels are applied to each detection. These size labels are based on the size of the bounding box detected, are different for each map scale, and can be changed in get_size_label.py.

   Then if lat/lon values are required, run again with the ``-c`` argument instead of ``-i``.
   - For 1:500 scale the location is set as the centre of the tree symbol.
   - For 1:2500 scale the location is set as the base of the tree symbol.

    It may be necessary to have ``gdal`` installed in separate environment to run this script as there may be conflicts with ``pytorch``. The code is setup to allow the conversion of coords without ``pytorch`` and vice-versa.

3. If running for one sheet or you are wanting to do each stage separtley (for one sheet only) then each script must be run individually. As long as the information has been set up correctly in ``config.py``, then run in order: ``split_image.py``, ``inference.py``, ``nms.py``, ``gt_size_label.py`` and ``convert_pixels_to_coord.py``.

4. Following these steps you will end up with several output csvs.
  
   They will be name as follows:
   - ``{map_name}_tree_coords.csv`` - contains the initial detections
   - ``{map_name}_tree_coords_nms.csv`` - contains the reduced detections, following NMS
   - ``{map_name}_tree_coords_size.csv`` - contains the detections following NMS, with a size label
   - ``{map_name}_tree_coords_lat_lon.csv`` - contains the detections following NMS, with a size label and with coordinate information. 

   The important columns to know about are:

   - score - the score (confidence) of the detection.
   - class - if the detection is classed as a tree (broadleaf) or conifer.
   - area - the area of the detected bounding box.
   - size - the size label applied to the detection.
   - lat - latitude of tree location (WGS 84).
   - lon - longitude of tree location (WGS 84).
   - lat_bng - y value of location in British National Grid (OSGB36).
   - lon_bng - x value of location in British National Grid (OSGB36).