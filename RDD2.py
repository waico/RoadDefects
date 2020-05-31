# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:35:10 2020

@author: kotsoev
"""

import matplotlib

from IPython import embed

#%%
def detect(dir):
    # import lib
    
    import numpy as np
    
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    
    from matplotlib import pyplot as plt
    from PIL import Image
    
    from secondary_functions import label_map_util
    from secondary_functions import visualization_utils as vis_util
    
    import os
    # import random
    
    #%% secondury functions
    
    matplotlib.use("Agg")

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
              (im_height, im_width, 3)).astype(np.uint8)
    
    
    #%% prepare variables
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_MODEL = 'trainedModels/ssd_inception_inference_graph.pb' # 'trainedModels/ssd_mobilenet_RoadDamageDetector.pb' 
    
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './crack_label_map.pbtxt' #'trainedModels/crack_label_map.pbtxt'
    
    NUM_CLASSES = 8
    
    #%% load model
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    #%% loading label map
            
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    #%% detection
    
    IMAGE_SIZE = (24, 16)
    
    
    #hack_test = [dir + filename for filename in os.listdir(dir)]
    

    hack_test = [ dir ]
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in hack_test: 
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], 
                                                     feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    min_score_thresh=0.3,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                return image_np
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)

#%% test

#f = open("test.jpg", "rb")

#image_np = detect(f)
#f.close()

#from PIL import Image
#q = Image.fromarray(image_np)
#q.save("test_0.jpg")

#embed()

