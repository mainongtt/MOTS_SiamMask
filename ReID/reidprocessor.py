import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

import os
import sys
import time

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


class ReID(object):
    def __init__(self, model_name = 'fp16'):
        '''
        model_name in ['fp16', 'ft_net_dense', 'ft_ResNet50']
        '''
        if model_name not in ['fp16', 'ft_net_dense', 'ft_ResNet50']:
            raise ValueError('ReID model name not correct')

        model_path = os.path.join('ReID/models', model_name)
        sys.path.append(model_path)
        import model

        if model_name == 'fp16':
            self.net = model.ft_net(751, stride=1)
        elif model_name == 'ft_net_dense':
            self.net = model.ft_net_dense(751, stride=1)
        elif model_name == 'ft_ResNet50':
            self.net = model.ft_net_middle(751, stride=1)

        pretrain_path = os.path.join('ReID/models', model_name, 'net_last.pth')
        self.net.load_state_dict( torch.load(pretrain_path) )
        self.net.eval()

        self.net.classifier = nn.Sequential()    # Remove the classifier layer in reference situation

    def get_reid_feature(self, image):
        tensor_img = Variable( im_to_torch(image).unsqueeze(0) )
        return self.net(tensor_img)    # In tensor 
    



if __name__ == "__main__":
    myreider = ReID('fp16')
    input = Variable(torch.FloatTensor(1, 3, 256, 128))
    start_time = time.time()
    for i in range(100):
        output = myreider.net(input)
        print(i)
    end_time = time.time()

