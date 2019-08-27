import os
import sys
SiamMaskPath = os.path.join(os.getcwd(), 'SiamMask')
sys.path.append(SiamMaskPath)

import cv2
import numpy as np
import singleTracker as singleTracker

if __name__ == '__main__':
    mytracking = singleTracker.SingleTracking()
    img1 = cv2.imread('SiamMask/testdata/img/00000001.jpg')
    target_pos = np.array([365, 194])
    target_sz = np.array([90, 120])

    examplar_feature = mytracking.get_examplar_feature(img1, target_pos, target_sz)
    

    for index in range(2,197):
        str_index = "%04d" % index
        img = cv2.imread('SiamMask/testdata/img/0000' + str_index + '.jpg')
        target_pos, target_sz, _, mask = mytracking.siamese_track(img, target_pos, target_sz, examplar_feature)

        mask = mask > 0.4
        img[:, :, 2] = mask * 255 + (1 - mask) * img[:, :, 2]
        cv2.imshow("result", img)
        cv2.waitKey(1)