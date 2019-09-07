###################################################
# MOTS_SiamMask
# Author:       Yunyao Mao
# Date update:  2019.09.07
# Email:        myy2016@mail.ustc.edu.cn
###################################################
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
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
    img = to_torch(img).float() / 255   # 0~255  to  0~1
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
        torch_img = im_to_torch(image)
        torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img).unsqueeze(0)
        input_tensor = Variable( torch_img )
        return self.net(input_tensor)    # In tensor 
    



if __name__ == "__main__":
    myreider = ReID('fp16')
    a = np.random.rand(256, 128, 3)
    torch_a = im_to_torch(a)
    torch_a = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_a).unsqueeze(0)
    print(torch_a)
    input = Variable(torch_a)

    output = myreider.net(input)
    print(output)


