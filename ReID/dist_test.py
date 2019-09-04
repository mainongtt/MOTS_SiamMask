import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import cv2

import os
import sys

class Market1501_Dataset(Dataset):
    def __init__(self, root_dir='Market', train=True, transform=None):
        '''
        Args:
            root_dir: (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        '''
        self.root_dir = root_dir
        if train == True:
            self.image_dirName = 'bounding_box_train'
        else:
            self.image_dirName = 'bounding_box_test'

        def get_image_list(dir):
            file_list = os.listdir(dir)
            return [item for item in file_list if 'jpg' in item]    # Filter the non-image element
        
        self.image_list = get_image_list( os.path.join(self.root_dir, self.image_dirName) )
        self.image_list.sort()

        self.transform = transform
        self.image_shape = (128, 256)    # Shape input to reid model
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        imageName = self.image_list[index]
        imagePath = os.path.join(self.root_dir, self.image_dirName, imageName)
        image = cv2.imread(imagePath, )
        image = cv2.resize(image, self.image_shape, interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int( imageName[0:4] )

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}



if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('Usage python dist_test.py model_name')
        exit(0)
    
    model_name = sys.argv[1]
    model_path = os.path.join('models', model_name)
    sys.path.append(model_path)
    import model

    if model_name == 'fp16':
        net = model.ft_net(751, stride=1)
    elif model_name == 'ft_net_dense':
        net = model.ft_net_dense(751, stride=1)
    elif model_name == 'ft_ResNet50':
        net = model.ft_net_middle(751, stride=1)

    pretrain_path = os.path.join('models', model_name, 'net_last.pth')
    net.load_state_dict( torch.load(pretrain_path) )
    net.eval()

    net.classifier = nn.Sequential()

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    my_dataset = Market1501_Dataset(transform=data_transform)

    dataset_loader = torch.utils.data.DataLoader(my_dataset,
                                                batch_size=2,
                                                shuffle=False,
                                                num_workers=4)
    
    for batch in dataset_loader:
        feature_1 = net(batch['image'])[ [0] ]
        feature_2 = net(batch['image'])[ [1] ]
        dist = torch.dist(feature_1, feature_2).item()
        if( batch['label'][0] == batch['label'][1] ):
            print(dist)