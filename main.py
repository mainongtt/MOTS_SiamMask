import os
import sys
SiamMaskPath = os.path.join(os.getcwd(), 'SiamMask')
sys.path.append(SiamMaskPath)
MaskRCNNPath = os.path.join(os.getcwd(), 'MaskRCNN')
sys.path.append(MaskRCNNPath)
ReIDPath = os.path.join(os.getcwd(), 'ReID')
sys.path.append(ReIDPath)


import cv2
import numpy as np
import PIL.Image as Image

from sklearn.utils.linear_assignment_ import linear_assignment

# Issue: 同时导入时会出bug
import singletracker
import reidprocessor
#import detector





## Args class for debugging only
class Args(object):
    def __init__(self):
        self.visualize = True
        self.siammask_threshold = 0.3
        self.iou_threshold = 0.3
        self.store_for_eval = True
        self.croped_obj_image_shape = (256, 128)
        self.score_threshold = 0.9




class Tracklet(object):
    def __init__(self, target_track_id, target_class_id, target_pos, target_sz, target_mask, target_score, examplar_feature, match_feature = None):
        self.target_track_id = target_track_id

        self.target_class_id = target_class_id
        if target_class_id == 1:
            self.base_number = 2000
        else:
            self.base_number = 1000
        
        self.target_pos = target_pos
        self.target_sz = target_sz
        self.target_mask = target_mask
        self.target_score = target_score
        self.examplar_feature = examplar_feature
        self.match_feature = match_feature    # To be defined

        self.update_flag = True    # Indicate that the tracklet is updated in current frame

        ## Parameter predicted by Siammask 
        self.predicted_pos = None
        self.predicted_sz = None
        self.predicted_mask = None
        self.predicted_score = None
    
    def update_state(self, target_pos, target_sz, target_mask, target_score, match_feature = None):
        self.target_pos = target_pos
        self.target_sz = target_sz
        self.target_mask = target_mask
        self.target_score = target_score
        self.match_feature = match_feature
        
        self.update_flag = True    # To prove the tracklet is updated in current frame


    def update_predicted_state(self, predicted_pos, predicted_sz, predicted_mask, predicted_score):
        self.predicted_pos = predicted_pos
        self.predicted_sz = predicted_sz
        self.predicted_mask = predicted_mask
        self.predicted_score = predicted_score    




def mask_iou(det_mask, pred_mask):
    '''
    Computes IoU between two masks
    Input: two 2D array mask
    '''
    Union = (pred_mask + det_mask) != 0
    Intersection =  (pred_mask * det_mask) != 0
    return np.sum(Intersection) / np.sum(Union)




def get_obj_croped_image(frame_image, obj_pos, obj_sz, output_shape):
    top_left_x = int( obj_pos[0] - obj_sz[0] / 2 )
    top_left_y  = int( obj_pos[1] - obj_sz[1] / 2 )
    bottom_right_x = int( top_left_x + obj_sz[0] )
    bottom_right_y = int( top_left_y + obj_sz[1] )
    crop_image = frame_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]
    
    assert (crop_image.shape[0] != 0 and crop_image.shape[1] != 0)
    return cv2.resize(crop_image, output_shape, interpolation = cv2.INTER_AREA)




def associate_detection_to_tracklets(det_result, tracklets, iou_threshold = 0.5):
    ## Conduct association between frame_detect_result and tracklets' predicted result
     # Without appearance matching 20190830
    frame_masks = det_result['masks']
    frame_rois = det_result['rois']
    frame_class_ids = det_result['class_ids']
    frame_scores = det_result['scores']

    tracklet_num = len(tracklets)
    det_object_num = frame_masks.shape[2]
    iou_matrix = np.zeros( shape=(det_object_num, tracklet_num), dtype=np.float32 )
    for det_object_index in range(det_object_num):
        for tracklet_index in range(tracklet_num):
            iou_matrix[det_object_index][tracklet_index] = mask_iou( frame_masks[:, :, det_object_index], tracklets[tracklet_index].predicted_mask )
    matched_indices = linear_assignment(-iou_matrix)


    unmatched_detections = []
    for det_object_index in range(det_object_num):
        if( det_object_index not in matched_indices[:,0] ):
            unmatched_detections.append(det_object_index)
    
    unmatched_tracklets = []
    for tracklet_index in range(tracklet_num):
        if( tracklet_index not in matched_indices[:,1] ):
            unmatched_tracklets.append(tracklet_index)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0],m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_tracklets.append(m[1])
        else:
            matches.append( m.reshape(1,2) )
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracklets)




def visualize_current_frame(frame_image, tracklets, pred=True):
    if pred == True:
        for tracklet in tracklets:
            if tracklet.predicted_mask is not None:
                mask = tracklet.predicted_mask
                frame_image[:, :, 2] = mask * 255 + (1 - mask) * frame_image[:, :, 2]
                text = 'id: ' + str(tracklet.target_track_id)
                cols = int(tracklet.predicted_pos[0])
                rows = int(tracklet.predicted_pos[1])
                cv2.putText(frame_image, text, (cols, rows), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)
    
    else:
        for tracklet in tracklets:
            mask = tracklet.target_mask
            frame_image[:, :, 2] = mask * 255 + (1 - mask) * frame_image[:, :, 2]
            text = 'id: ' + str(tracklet.target_track_id)
            cols = int(tracklet.target_pos[0])
            rows = int(tracklet.target_pos[1])
            cv2.putText(frame_image, text, (cols, rows), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)
    return frame_image




def frame_store_for_eval(frame_image, tracklets):
    tracklet_num = len(tracklets)
    if tracklet_num == 0:
        return np.zeros(shape = frame_image.shape[0:2]).astype(np.int32)
    
    mask_list = [ (tracklet.target_track_id + tracklet.base_number) * tracklet.target_mask for tracklet in tracklets ]

    ## Here is just a simple method to solve overlap
    result = mask_list[0]
    for index in range(1, tracklet_num):
        overlap_mask = result * mask_list[index] == 0
        result += mask_list[index] * overlap_mask
    return result.astype(np.int32)






if __name__ == '__main__':
    
    args = Args()    # Debug args

    
    ## Single object tracker Siammask
    ## Defined in SiamMask/singletracker.py
    vot_model_path = 'SiamMask/pretrained/SiamMask_VOT.pth'
    vot_config_path = 'SiamMask/config/config_vot.json'
    mytracker = singletracker.SingleTracker(vot_config_path, vot_model_path)
    myreider = reidprocessor.ReID('fp16')
    

    ## Mian process pipeline
    dataset_path = 'Dataset/MOTSChallenge'
    det_result_path = os.path.join(dataset_path, 'maskrcnn')
    track_result_path = 'Result'
    if not os.path.exists(track_result_path):
        os.mkdir(track_result_path)
    
    videos = os.listdir(det_result_path)    #['0002', '0005', ...]
    for video in videos:
        video_path = os.path.join(det_result_path, video)
        video_track_result_path = os.path.join(track_result_path, video)
        if not os.path.exists(video_track_result_path):
            os.mkdir(video_track_result_path)
        
        frames = os.listdir(video_path)    #['000001.npz', '000002.npz',...]
        frames.sort()    # To make the frames in order

        #### Tracking for a video start
        tracklets = []    # A list to store tracklets for a video
        track_id_to_assign = 0    # The unused track_id to be assigned

        for frame in frames:
            raw_image_path = os.path.join(video_path, frame).replace('maskrcnn', 'images').replace('npz', 'jpg')
            frame_image = cv2.imread(raw_image_path)

            det_result = np.load( os.path.join(video_path, frame) )

            ## Select the detection result with class people (1) and car (3)
            class_id_set = set( [1, 3] )
            class_filter = [ index for index, item in enumerate(det_result['class_ids']) if item in class_id_set]
            score_filter = [ index for index, item in enumerate(det_result['scores']) if item > args.score_threshold]
            det_filter = [index for index in class_filter if index in score_filter]

            filt_det_result = {}
            filt_det_result['masks'] = det_result['masks'][:, :, det_filter]
            filt_det_result['rois'] = det_result['rois'][det_filter]
            filt_det_result['class_ids'] = det_result['class_ids'][det_filter]
            filt_det_result['scores'] = det_result['scores'][det_filter]

            det_result = filt_det_result


            for tracklet in tracklets:
                tracklet.update_flag = False

            if len(tracklets) == 0:
                ## Init trackldet_object_numets with current frame
                frame_masks = det_result['masks']
                frame_rois = det_result['rois']
                frame_class_ids = det_result['class_ids']
                frame_scores = det_result['scores']
                det_object_num = frame_masks.shape[2]

                for obj_index in range(det_object_num):
                    
                    obj_class_id = frame_class_ids[obj_index]

                    obj_roi = frame_rois[obj_index]

                    obj_pos = np.array( [np.mean(obj_roi[1::2]), np.mean(obj_roi[0::2])] )
                    obj_sz = np.array( [obj_roi[3]-obj_roi[1], obj_roi[2]-obj_roi[0]] )
                    
                    obj_mask = frame_masks[:, :, obj_index]
                    obj_score = frame_scores[obj_index]
                    obj_croped_image = get_obj_croped_image(frame_image, obj_pos, obj_sz, args.croped_obj_image_shape)


                    examplar_feature = mytracker.get_examplar_feature(frame_image, obj_pos, obj_sz)
                    match_feature = myreider.get_reid_feature(obj_croped_image)
                    
                    tracklet = Tracklet(track_id_to_assign, 
                                        obj_class_id, 
                                        obj_pos, 
                                        obj_sz, 
                                        obj_mask, 
                                        obj_score, 
                                        examplar_feature, 
                                        match_feature)
                    track_id_to_assign += 1    # Increase the unused track id 
                    tracklets.append(tracklet)
            
            else:
                for tracklet in tracklets:
                    predicted_result = mytracker.siamese_track( frame_image,
                                                                tracklet.target_pos,
                                                                tracklet.target_sz,
                                                                tracklet.examplar_feature)
                    predicted_pos, predicted_sz, predicted_score, predicted_mask = predicted_result
                    predicted_mask = predicted_mask > args.siammask_threshold    # To get a binary mask

                    tracklet.update_predicted_state(predicted_pos, predicted_sz, predicted_mask, predicted_score)
                
                matched, unmatched_det_result, unmatched_tracklets = associate_detection_to_tracklets(det_result, tracklets, args.iou_threshold)
                
                ## Update matched tracklets with assigned det result
                for tracklet_index, tracklet in enumerate(tracklets):
                    if (tracklet_index not in unmatched_tracklets):
                        det_result_index = int( matched[np.where(matched[:, 1]==tracklet_index)[0], 0] )    # det_result_index have to be a value not an array

                        obj_roi = det_result['rois'][det_result_index]

                        obj_pos = np.array( [np.mean(obj_roi[1::2]), np.mean(obj_roi[0::2])] )
                        obj_sz = np.array( [obj_roi[3]-obj_roi[1], obj_roi[2]-obj_roi[0]] )

                        obj_mask = det_result['masks'][:, :, det_result_index]
                        obj_score = det_result['scores'][det_result_index]

                        obj_croped_image = get_obj_croped_image(frame_image, tracklet.target_pos, tracklet.target_sz, args.croped_obj_image_shape)
                        match_feature = myreider.get_reid_feature(obj_croped_image)

                        tracklet.update_state(obj_pos, obj_sz, obj_mask, obj_score, match_feature)

                ## Create and initialise new tracklets for unmatched det result
                for det_result_index in unmatched_det_result:
                    obj_class_id = det_result['class_ids'][det_result_index]

                    obj_roi = det_result['rois'][det_result_index]
                    obj_pos = np.array( [np.mean(obj_roi[1::2]), np.mean(obj_roi[0::2])] )    # rois = np.array([y1, x1, y2, x2]) , obj_pos = np.array([x, y])
                    obj_sz = np.array( [obj_roi[3]-obj_roi[1], obj_roi[2]-obj_roi[0]] )    # rois = np.array([y1, x1, y2, x2]) , obj_sz = np.array([width, height])

                    obj_mask = det_result['masks'][:, :, det_result_index]
                    obj_score = det_result['scores'][det_result_index]
                    
                    obj_croped_image = get_obj_croped_image(frame_image, obj_pos, obj_sz, args.croped_obj_image_shape)
                    
                    examplar_feature = mytracker.get_examplar_feature(frame_image, obj_pos, obj_sz)
                    match_feature = myreider.get_reid_feature(obj_croped_image)
                    
                    tracklet = Tracklet(track_id_to_assign, 
                                        obj_class_id, 
                                        obj_pos, 
                                        obj_sz, 
                                        obj_mask, 
                                        obj_score, 
                                        examplar_feature, 
                                        match_feature)
                    track_id_to_assign += 1    # Increase the unused track id 
                    tracklets.append(tracklet)
                
                ## Remove untracked tracklets
                tracklet_index = len(tracklets)
                for tracklet in reversed(tracklets):
                    tracklet_index -= 1
                    if tracklet.update_flag == False:
                        tracklets.pop(tracklet_index)

            if args.visualize == True:
                visual = visualize_current_frame(frame_image, tracklets, pred=False)
                visual_save_path = os.path.join(video_track_result_path, frame.split('.')[0] + '.jpg')
                cv2.imwrite(visual_save_path, visual)

            if args.store_for_eval == True:
                eval_array = frame_store_for_eval(frame_image, tracklets)
                eval_save_path = os.path.join(video_track_result_path, frame.split('.')[0] + '.png')
                eval_png = Image.fromarray(eval_array)
                eval_png.save(eval_save_path)
            