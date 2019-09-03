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
        Return format:
            [{'masks': array with shape = (image_h, image_w, object_num)
              'rois': array with shape = (object_num, 4) where 4 indicate: y1, x1, y2, x2
              'class_ids': array with shape = (object_num,)
              'scores': array with shape = (object_num,)}, 
             {'masks': array with shape = (image_h, image_w, object_num)
              'rois': array with shape = (object_num, 4)
              'class_ids': array with shape = (object_num,)
              'scores': array with shape = (object_num,)},
             {'masks': array with shape = (image_h, image_w, object_num)
              'rois': array with shape = (object_num, 4)
              'class_ids': array with shape = (object_num,)
              'scores': array with shape = (object_num,)},
             ......
             ......
            ]
        '''
        result = self.model.detect(imgs, verbose=verbose)
        return result




if __name__ == "__main__":
    coco_model_path = 'pretrained/mask_rcnn_coco.h5'
    model_dir = 'logs'
    mydetector = Detector(coco_model_path, model_dir)

    dataset_path = '../Dataset/KITTYMOTS'
    videos_path = os.path.join(dataset_path, 'images')
    videos = os.listdir(videos_path)   #['0002', '0005', ...]

    det_result_path = os.path.join(dataset_path, 'maskrcnn')
    if not os.path.exists(det_result_path):
        os.mkdir(det_result_path)
    
    for video in videos:
        video_path = os.path.join(videos_path, video)

        output_path = os.path.join(det_result_path, video)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        frames = os.listdir(video_path) #['000001.jpg', '000002.jpg',...]
        

        for frame in frames:
            img = skimage.io.imread(os.path.join(video_path, frame))
            det_result = mydetector.detect([img])

            frame_masks = det_result[0]['masks']
            frame_rois = det_result[0]['rois']
            frame_class_ids = det_result[0]['class_ids']
            frame_scores = det_result[0]['scores']

            frame_without_suffix = frame.split('.')[0]
            np.savez_compressed(os.path.join(output_path, frame_without_suffix + '.npz'), 
                                    masks=frame_masks,
                                    rois=frame_rois,
                                    class_ids=frame_class_ids,
                                    scores=frame_scores )
            