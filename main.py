import os
import sys
SiamMaskPath = os.path.join(os.getcwd(), 'SiamMask')
sys.path.append(SiamMaskPath)

import cv2
import numpy as np
import singleTracker as singleTracker



class Tracklet(object):
    def __init__(self, target_id, target_pos, target_sz, target_mask, target_score, examplar_feature):
        self.target_id = target_id
        self.examplar_feature = examplar_feature

        self.target_pos = target_pos
        self.target_sz = target_sz
        self.target_mask = target_mask
        self.target_score = target_score
    
    def update_state(self, target_pos, target_sz, target_mask, target_score):
        self.target_pos = target_pos
        self.target_sz = target_sz
        self.target_mask = target_mask
        self.target_score = target_score






if __name__ == '__main__':

    mytracking = singleTracker.SingleTracking()

    tracklets = []

    




    
    img1 = cv2.imread('SiamMask/testdata/img/000000.png')
    print(img1.shape)
    target_pos = np.array([813, 281.25])    # target_pos: np.array([cols, rows]) which indicate the center point position
    target_sz = np.array([95, 187.5])       # target_sz:  np.array([target_width, target_height]) which indicate the target size

    examplar_feature = mytracking.get_examplar_feature(img1, target_pos, target_sz)
    

    for index in range(154):
        str_index = "%04d" % index
        img = cv2.imread('SiamMask/testdata/img/00' + str_index + '.png')
        target_pos, target_sz, _, mask = mytracking.siamese_track(img, target_pos, target_sz, examplar_feature)

        mask = mask > 0.3
        img[:, :, 2] = mask * 255 + (1 - mask) * img[:, :, 2]
        cv2.imshow("result", img)
        cv2.waitKey(1)