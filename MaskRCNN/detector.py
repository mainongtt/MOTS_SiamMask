import os
import sys
import random
import math
import numpy as np
import skimage.io
#import matplotlib
#import matplotlib.pyplot as plt


# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import samples.coco.coco as coco


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1





class Detector(object):
    def __init__(self, coco_model_path, model_dir):
        # Local path to trained weights file
        # When called by main.py
        self.COCO_MODEL_PATH = coco_model_path  # "MaskRCNN/pretrained/mask_rcnn_coco.h5"
        if not os.path.exists(self.COCO_MODEL_PATH):
            raise ValueError("Please download pretrained model (mask_rcnn_coco.h5) first")
        # Directory to save logs and trained model
        # When called by main.py
        self.MODEL_DIR = model_dir  # "MaskRCNN/logs"

        self.config = InferenceConfig()
        #self.config.display()

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                            'bus', 'train', 'truck', 'boat', 'traffic light',
                            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'b|ear',
                            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                            'kite', 'baseball bat', 'baseball glove', 'skateboard',
                            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                            'teddy bear', 'hair drier', 'toothbrush']

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)

        # Load weights trained on MS-COCO
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)

    def detect(self, imgs, verbose=1):
        '''
        Params:
            imgs: a list of images like [img1,img2,...]
            verbose:
        '''
        result = self.model.detect(imgs, verbose=verbose)
        return result




if __name__ == "__main__":
    coco_model_path = 'pretrained/mask_rcnn_coco.h5'
    model_dir = 'logs'
    mydetector = Detector(coco_model_path, model_dir)

    img = skimage.io.imread('images/9247489789_132c0d534a_z.jpg')
    for i in range(4):
        result = mydetector.detect([img])
    print(result[0]['masks'].shape)