# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
import argparse
import logging
import numpy as np
import cv2
from PIL import Image
from os import makedirs
from os.path import join, isdir, isfile

from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig

from utils.config_helper import load_config
from utils.pyvotkit.region import vot_overlap, vot_float2str


thrs = np.arange(0.3, 0.5, 0.05)

parser = argparse.ArgumentParser(description='SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('--config', default='SiamMask/config/config_vot.json', dest='config', help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='SiamMask/pretrained/SiamMask_VOT.pth', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--mask', default=True, action='store_true', help='whether use mask output')
parser.add_argument('--refine', default=True, action='store_true', help='whether use mask refine output')
parser.add_argument('--cpu', action='store_true', help='cpu mode')


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    # cv2.imshow('crop', im_patch)
    # cv2.waitKey(0)
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch


def generate_anchor(cfg, score_size):
    anchors = Anchors(cfg)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)

    total_stride = anchors.stride
    anchor_num = anchor.shape[0]

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def MultiBatchIouMeter(thrs, outputs, targets, start=None, end=None):
    targets = np.array(targets)
    outputs = np.array(outputs)

    num_frame = targets.shape[0]
    if start is None:
        object_ids = np.array(list(range(outputs.shape[0]))) + 1
    else:
        object_ids = [int(id) for id in start]

    num_object = len(object_ids)
    res = np.zeros((num_object, len(thrs)), dtype=np.float32)

    output_max_id = np.argmax(outputs, axis=0).astype('uint8')+1
    outputs_max = np.max(outputs, axis=0)
    for k, thr in enumerate(thrs):
        output_thr = outputs_max > thr
        for j in range(num_object):
            target_j = targets == object_ids[j]

            if start is None:
                start_frame, end_frame = 1, num_frame - 1
            else:
                start_frame, end_frame = start[str(object_ids[j])] + 1, end[str(object_ids[j])] - 1
            iou = []
            for i in range(start_frame, end_frame):
                pred = (output_thr[i] * output_max_id[i]) == (j+1)
                mask_sum = (pred == 1).astype(np.uint8) + (target_j[i] > 0).astype(np.uint8)
                intxn = np.sum(mask_sum == 2)
                union = np.sum(mask_sum > 0)
                if union > 0:
                    iou.append(intxn / union)
                elif union == 0 and intxn == 0:
                    iou.append(1)
            res[j, k] = np.mean(iou)
    return res



########################################################################
# Author:            myy
# Date:              20180828
# Description:       修改了 Siammask 的测试代码得到一个单目标跟踪的 Trcker
#                    尚未对 Siammask 的具体的一些超参数进行细致理解
#                    Dangerous 中的代码最好不要动
########################################################################
class SingleTracking(object):
    def __init__(self):
        args = parser.parse_args()
        cfg = load_config(args)
        if args.arch == 'Custom':
            from custom import Custom
            self.model = Custom(anchors=cfg['anchors'])
        else:
            parser.error('invalid architecture: {}'.format(args.arch))

        if args.resume:
            assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
            self.model = load_pretrain(self.model, args.resume)
        self.model.eval()
        self.device = torch.device('cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
        self.model = self.model.to(self.device)

        ################# Dangerous
        self.p = TrackerConfig()
        self.p.update(cfg['hp'] if 'hp' in cfg.keys() else None, self.model.anchors)
        self.p.renew()

        self.p.scales = self.model.anchors['scales']
        self.p.ratios = self.model.anchors['ratios']
        self.p.anchor_num = self.model.anchor_num
        self.p.anchor = generate_anchor(self.model.anchors, self.p.score_size)

        if self.p.windowing == 'cosine':
            self.window = np.outer(np.hanning(self.p.score_size), np.hanning(self.p.score_size))
        elif self.p.windowing == 'uniform':
            self.window = np.ones((self.p.score_size, self.p.score_size))
        self.window = np.tile(self.window.flatten(), self.p.anchor_num)
        ################


    def get_examplar_feature(self, img, target_pos, target_sz):
        avg_chans = np.mean(img, axis=(0, 1))

        wc_z = target_sz[0] + self.p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + self.p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        # initialize the exemplar
        examplar = get_subwindow_tracking(img, target_pos, self.p.exemplar_size, s_z, avg_chans)

        z = Variable(examplar.unsqueeze(0))
        return self.model.template(z.to(self.device))

    def siamese_track(self, img, target_pos, target_sz, examplar_feature, debug=False, mask_enable=True, refine_enable=True):
        avg_chans = np.mean(img, axis=(0, 1))
        im_h = img.shape[0]
        im_w = img.shape[1]

        wc_x = target_sz[1] + self.p.context_amount * sum(target_sz)
        hc_x = target_sz[0] + self.p.context_amount * sum(target_sz)
        s_x = np.sqrt(wc_x * hc_x)
        scale_x = self.p.exemplar_size / s_x
        d_search = (self.p.instance_size - self.p.exemplar_size) / 2
        pad = d_search / scale_x
        s_x = s_x + 2 * pad
        crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]
        
        # extract scaled crops for search region x at previous target position
        x_crop = Variable(get_subwindow_tracking(img, target_pos, self.p.instance_size, round(s_x), avg_chans).unsqueeze(0))

        if mask_enable:
            score, delta, mask = self.model.track_mask(examplar_feature, x_crop.to(self.device))
        else:
            score, delta = self.model.track(examplar_feature, x_crop.to(self.device))

        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
        score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
                1].cpu().numpy()

        delta[0, :] = delta[0, :] * self.p.anchor[:, 2] + self.p.anchor[:, 0]
        delta[1, :] = delta[1, :] * self.p.anchor[:, 3] + self.p.anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * self.p.anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * self.p.anchor[:, 3]

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        target_sz_in_crop = target_sz*scale_x
        s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
        r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * self.p.penalty_k)
        pscore = penalty * score

        # cos window (motion model)
        pscore = pscore * (1 - self.p.window_influence) + self.window * self.p.window_influence
        best_pscore_id = np.argmax(pscore)

        pred_in_crop = delta[:, best_pscore_id] / scale_x
        lr = penalty[best_pscore_id] * score[best_pscore_id] * self.p.lr  # lr for OTB

        res_x = pred_in_crop[0] + target_pos[0]
        res_y = pred_in_crop[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])

        # for Mask Branch
        if mask_enable:
            best_pscore_id_mask = np.unravel_index(best_pscore_id, (5, self.p.score_size, self.p.score_size))
            delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]

            if refine_enable:
                mask = self.model.track_refine((delta_y, delta_x)).to(self.device).sigmoid().squeeze().view(
                    self.p.out_size, self.p.out_size).cpu().data.numpy()
            else:
                mask = mask[0, :, delta_y, delta_x].sigmoid(). \
                    squeeze().view(self.p.out_size, self.p.out_size).cpu().data.numpy()

            def crop_back(image, bbox, out_sz, padding=-1):
                a = (out_sz[0] - 1) / bbox[2]
                b = (out_sz[1] - 1) / bbox[3]
                c = -a * bbox[0]
                d = -b * bbox[1]
                mapping = np.array([[a, 0, c],
                                    [0, b, d]]).astype(np.float)
                crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=padding)
                return crop

            s = crop_box[2] / self.p.instance_size
            sub_box = [crop_box[0] + (delta_x - self.p.base_size / 2) * self.p.total_stride * s,
                    crop_box[1] + (delta_y - self.p.base_size / 2) * self.p.total_stride * s,
                    s * self.p.exemplar_size, s * self.p.exemplar_size]
            s = self.p.out_size / sub_box[2]
            back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w * s, im_h * s]
            mask_in_img = crop_back(mask, back_box, (im_w, im_h))

            target_mask = (mask_in_img > self.p.seg_thr).astype(np.uint8)
            if cv2.__version__[-5] == '4':
                contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            else:
                _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt_area = [cv2.contourArea(cnt) for cnt in contours]
            if len(contours) != 0 and np.max(cnt_area) > 100:
                contour = contours[np.argmax(cnt_area)]  # use max area polygon
                polygon = contour.reshape(-1, 2)
                # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
                prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle

                # box_in_img = pbox
                rbox_in_img = prbox
            else:  # empty mask
                location = cxy_wh_2_rect(target_pos, target_sz)
                rbox_in_img = np.array([[location[0], location[1]],
                                        [location[0] + location[2], location[1]],
                                        [location[0] + location[2], location[1] + location[3]],
                                        [location[0], location[1] + location[3]]])

        target_pos[0] = max(0, min(im_w, target_pos[0]))
        target_pos[1] = max(0, min(im_h, target_pos[1]))
        target_sz[0] = max(10, min(im_w, target_sz[0]))
        target_sz[1] = max(10, min(im_h, target_sz[1]))

        score = score[best_pscore_id]
        mask = mask_in_img if mask_enable else []
        return target_pos, target_sz, score, mask

    


if __name__ == '__main__':
    mytracking = SingleTracking()
    img1 = cv2.imread('testdata/img/00000001.jpg')
    img2 = cv2.imread('testdata/img/00000002.jpg')
    target_pos = np.array([365, 194])
    target_sz = np.array([90, 120])

    examplar_feature = mytracking.get_examplar_feature(img1, target_pos, target_sz)
    _, _, _, mask = mytracking.siamese_track(img, target_pos, target_sz, examplar_feature)
    cv2.imshow("test", mask)
    cv2.waitKey(0)


