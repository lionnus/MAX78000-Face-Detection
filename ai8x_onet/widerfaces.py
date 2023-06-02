###################################################################################################
# WIDER Faces dataloader
# Lionnus Kesting
# Machine Learning on Microcontrollers
# 2023 - ETH Zurich
###################################################################################################
"""
WIDER Faces dataset
"""
import os
import cv2

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import random

import ai8x

"""
Custom image dataset class
"""
class WIDERFacesDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.annotation_file = os.path.join(self.data_path, "wider_face_bbx_gt_no_faces.txt")
        self.transform = transform

        self.flags = {	
                    # 'blur': 0,	
                    # 'expression': 0,	
                    # 'illumination': 0,	
                    # 'invalid': 0,	
                    'occlusion': 0,	
                    'pose': 0	
                }
        
        # Load data and annotations
        self.data, self.annotations = self.load_data()
        
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.annotations[idx]['image'])
        image = self.load_image(image_path)
        
        bboxes = self.annotations[idx]['bboxes']
        face_label = self.annotations[idx]['labels']['faces']
        # Resize image and bbox if the face is too small	
        image, bboxes = self.square_crop_around_bbox(image, bboxes[0])

        # Resize image to 48 x 48 and update boundary boxes accordingly
        bboxes_resized = self.resize_bbox(bboxes, image.shape[1], image.shape[0], 48,48)
        image = cv2.resize(image, (48,48))

        # Convert bboxes to floats and normalize to [0,1]
        bboxes = [float(i)/48 for i in bboxes_resized[0]]
        
        # Create face (class) labels with way too complicated logic
        one_hot_face_label = [0,0]
        one_hot_face_label[0] = float(face_label == 0) # [1,0] for face_label == 1
        one_hot_face_label[1] = float(face_label == 1) # [0,1] for face_label == 0
        one_hot_face_label = torch.tensor([one_hot_face_label])

        # Target tensor with botch bboxes and face labels
        target = torch.tensor(bboxes + [float(face_label)])
        
        if self.transform:
            image = self.transform(image)

        return image, target
    
    def load_data(self):
        annotations = []
        data = []
        
        image_files = os.listdir(self.data_path) # list of all the objects in directory, also includes the folder->check for extension to get images

        with open(self.annotation_file, 'r') as f:
            lines = f.read().splitlines()

        i = 0
        while(i<len(lines)):
                image_name = lines[i]
                image_file_path = os.path.join(self.data_path, image_name)
                i += 1
                num_bboxes = int(lines[i])

                bboxes = []

                # Iterate over the lines containing the boundary box coordinates
                for j in range(num_bboxes):
                    i += 1
                    bbox_data = lines[i].split(' ')
                    bbox = [
                        int(bbox_data[0]),
                        int(bbox_data[1]),
                        int(bbox_data[2]),
                        int(bbox_data[3])
                    ]
                    bboxes.append(bbox)
                # Fix the stupid fact that it has 0 coordinates if it doesnt have a boundary box
                if(num_bboxes==0):
                    bbox_data = [0,0,0,0,0,0,0,0,0,0]
                    bbox=[0,0,0,0]
                    bboxes.append(bbox)
                    i+=1
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
                 # Filter for images with only a single face	
                if(num_bboxes ==1 or num_bboxes==0):	
                    # Filter out images with have certain unusual flags	
                    if all(label[key] == 0 for key in self.flags.keys()):	
                        annotations.append(annotation)	
                        data.append(image_file_path)
                i += 1

        return data, annotations
    def square_crop_around_bbox(self, image, bbox, min_ratio=0.15, max_ratio=0.8):	
        image_height, image_width, _ = image.shape	
        bbox_x = bbox[0]	
        bbox_y = bbox[1]	
        bbox_width = bbox[2]	
        bbox_height = bbox[3]	
        # Dont do anything if the bbox is empty	
        if (bbox_width == 0 or bbox_height == 0):	
            return image, [bbox]	
            
        # Calculate the ratio of bbox width or height to the smallest image dimension	
        ratio = min(bbox_width / image_width, bbox_height / image_height)	
            
        # print("Image Dimensions: ", image_width, "x", image_height)	
        # print("Bounding Box Dimensions: ", bbox_width, "x", bbox_height)	
        # print("Bounding Box Ratio: ", ratio)	
        # Check if the ratio is smaller than the threshold	
        # if ratio < min_ratio:	
        #     print("Found image with ratio smaller than the minimum threshold: ", ratio)	
        proposed_width = random.randint(int(bbox_width / max_ratio), int(bbox_width / min_ratio))	
        proposed_height = random.randint(int(bbox_height / max_ratio), int(bbox_height / min_ratio))	
        # Make images square whilst we're at it	
        min_new_dim = min(proposed_width, proposed_height)	
        new_width = min_new_dim if min_new_dim > bbox_width else proposed_width	
        new_height = min_new_dim if min_new_dim > bbox_height else proposed_height	
        # Calculate the offsets	
        # print("Min x_offset: ", max(0,bbox_x+bbox_width-new_width), "Max x_offset: ", bbox_x-1)	
        x_offset = random.randint(max(0,bbox_x+bbox_width-new_width), bbox_x)	
        # print("Min y_offset: ", max(0,bbox_y+bbox_height-new_height), "Max y_offset: ", bbox_y-1)	
        y_offset = random.randint(max(0,bbox_y+bbox_height-new_height), bbox_y)	
        # print("Offsets: ", x_offset, y_offset)	
        # Perform the cropping operation	
        cropped_image = image[y_offset:y_offset + new_height, x_offset:x_offset + new_width]	
        bbox[0] -= x_offset	
        bbox[1] -= y_offset	
            
        # print("New minimum ratio: ", min(bbox_width / new_width, bbox_height / new_height))	
        return cropped_image, [bbox]

    def resize_bbox(self,bboxes, dim_x_init,dim_y_init, dim_x,dim_y):
        bboxes_resized = []
        #print(bboxes, dim_x_init,dim_y_init, dim_x,dim_y)
        for bbox in bboxes:
            # Calculate the scaling factors for width and height
            scale_x = dim_x / dim_x_init
            scale_y = dim_y / dim_y_init
            #print(scale_x,scale_y)
            # Convert the coordinates to the new dimensions
            bbox_resized = [ int(bbox[0] * scale_x), int(bbox[1] * scale_y), int(bbox[2] * scale_x), int(bbox[3] * scale_y)]
            bboxes_resized.append(bbox_resized)
        #print(bboxes_resized)

        return bboxes_resized
    
    def load_image(self, path):
        # Load and preprocess the image
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
def widerfaces_get_datasets(data, load_train=True, load_test=True):
    """
    Load the WIDER Faces dataset

    The images are of multiple sizes, so they are rescaled to a predefined size.
    """
    (data_dir, args) = data

    if load_train:
        print("Loading training dataset")
        train_transform = transforms.Compose([
            #Rescale(256),
            #RandomCrop(224),
            transforms.ToTensor(),
            transforms.ColorJitter(),
            ai8x.normalize(args=args)
        ])

        train_dataset = WIDERFacesDataset(data_path=os.path.join(data_dir, "widerface", "WIDER_train/images"), transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        print("Loading test dataset")
        test_transform = transforms.Compose([
            #Rescale(256),
            #RandomCrop(224),
            transforms.ToTensor(),
            transforms.ColorJitter(),
            ai8x.normalize(args=args)
        ])
        # Load validation dataset instead of test dataset, since test dataset is unlabeled
        test_dataset = WIDERFacesDataset(data_path=os.path.join(data_dir, "widerface", "WIDER_val/images"), transform=test_transform)
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'widerfaces',
        'input': (3, 48, 48),
        'output': [('x', float), ('y', float), ('w', float), ('h', float), ('c', float)],
        'regression': True,
        'loader': widerfaces_get_datasets,
    },
]