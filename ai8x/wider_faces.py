###################################################################################################
# WIDER Faces dataloader
# Lionnus Kesting
# Machine Learning on Microcontrollers
# 2023 - ETH Zurich
###################################################################################################
"""
WIDER Faces dataset
"""
# Start with the basics
import numpy as np
from pandas import DataFrame
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Library for plotting
import matplotlib.pyplot as plt

# pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.io import read_image


# Import library for opening .mat files
import scipy.io

# Import ai8x specifics
import ai8x

"""
Custom image dataset class
"""
class WIDERFacesDataset(Dataset):
    def __init__(self, data_path, annotation_file, transform=None):
        self.data_path = data_path
        self.annotation_file = annotation_file
        self.transform = transform
        
        # Load data and annotations
        self.data = self.load_data()
        self.annotations = self.load_annotations()
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.annotations[idx]['image'])
        image = self.load_image(image_path)
        
        bboxes = self.annotations[idx]['bboxes']
        labels = self.annotations[idx]['labels']
        
        if self.transform:
            image, bboxes, labels = self.transform(image, bboxes, labels)
            
        return image, bboxes, labels
    
    def load_data(self):
        data = []
        
        image_files = os.listdir(self.data_path)
        for image_file in image_files:
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                image_file_path = os.path.join(self.data_path, image_file)
                data.append(image_file_path)
        return data
    
    def load_annotations(self):
        annotations = []

        with open(self.annotation_file, 'r') as f:
            lines = f.read().splitlines()

        i = 0
        while i < len(lines):
            if lines[i].endswith('.jpg') or lines[i].endswith('.png'):
                image_name = lines[i]
                i += 1
                num_bboxes = int(lines[i])
                i += 1

                bboxes = []
                labels = []

                # Iterate over the lines containing the boundary box coordinates
                for j in range(num_bboxes):
                    bbox_data = lines[i].split(' ')
                    bbox = [
                        int(bbox_data[0]),
                        int(bbox_data[1]),
                        int(bbox_data[2]),
                        int(bbox_data[3])
                    ]
                    bboxes.append(bbox)
                    i += 1
                # Gather all the labels of the current image
                label = {
                    'name': image_name.split('/')[1],
                    'faces': num_bboxes,
                    'type': int(image_name.split("--")[0]),
                    'blur': int(bbox_data[4]),
                    'expression': int(bbox_data[5]),
                    'illumination': int(bbox_data[6]),
                    'invalid': int(bbox_data[7]),
                    'occlusion': int(bbox_data[8]),
                    'pose': int(bbox_data[9])
                }
                annotation = {
                    'image': image_name,
                    'bboxes': bboxes,
                    'labels': label
                }
                annotations.append(annotation)
            i += 1
        return annotations
    
    def load_image(self, path):
        # Load and preprocess the image
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # TODO: Add additional processes below 
        
        return image
"""
Dataloader function
"""
"""
Dataloader function
"""
def WIDER_faces_get_datasets(data, load_train=False, load_test=False, transforms=None):
   
    (data_dir, args) = data
    # data_dir = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=30, translate=(0.5, 0.5), scale=(0.5,1.5), fill=0),
            
            ############################
            # TODO: Add more transform #
            ############################
            
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            F.normalize(args=args)
        ])

        train_dataset = WIDERFacesDataset(img_dir=os.path.join(data_dir, "WIDER_faces", "train"), transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            # 960 and 720 are not random, but dimension of input test img
            transforms.CenterCrop((960,720)),
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            F.normalize(args=args)
        ])
        test_dataset = WIDERFacesDataset(img_dir=os.path.join(data_dir, "WIDER_faces", "test"), transform=test_transform)
    else:
        test_dataset = None

    return train_dataset, test_dataset


"""
Dataset description
"""
datasets = [
    {
        'name': 'WIDER_faces',
        'input': (3, None, None),  # Variable input size
        'output': list(map(str, range(4))),
        'loader': WIDER_faces_get_datasets,
    }
]