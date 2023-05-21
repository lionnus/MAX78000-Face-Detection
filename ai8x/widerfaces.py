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

import ai8x
"""
Custom image dataset class
"""
class WIDERFacesDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.annotation_file = os.path.join(self.data_path, "wider_face_bbx_gt.txt")
        self.transform = transform
        
        # Load data and annotations
        self.data, self.annotations = self.load_data()
        
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.annotations[idx]['image'])
        image = self.load_image(image_path)
        
        bboxes = self.annotations[idx]['bboxes']

        # Resize image to 128 x 128 and update boundary boxes accordingly
        bboxes_resized = self.resize_bbox(bboxes, image.shape[1], image.shape[0], 128, 128)
        image = cv2.resize(image, (128, 128))

        #labels = self.annotations[idx]['labels']
        
        if self.transform:
            image = self.transform(image)
        
            
        return image, bboxes_resized
    
    def load_data(self):
        annotations = []
        data = []
        
        image_files = os.listdir(self.data_path) # list of all the objects in directory, also includes the folder->check for extension to get images

        with open(self.annotation_file, 'r') as f:
            lines = f.read().splitlines()

        i = 0
        print('do we even arrive here')
        while(i<len(lines)):
                image_name = lines[i]
                print('image name',lines[i])
                image_file_path = os.path.join(self.data_path, image_name)
                i += 1
                print('number of bboxes=',lines[i])
                num_bboxes = int(lines[i])

                bboxes = []
                labels = []

                # Iterate over the lines containing the boundary box coordinates
                for j in range(num_bboxes):
                    i += 1
                    print('bboxx',lines[i])
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
                print(annotation['labels']['faces'])
                if(num_bboxes==1):
                    annotations.append(annotation)
                    data.append(image_file_path)
                    print('well, something has been added',lines[i])
                i += 1

        return data, annotations
    
    def resize_bbox(self,bboxes, dim_x_init,dim_y_init, dim_x,dim_y):
        bboxes_resized = []
        print(bboxes, dim_x_init,dim_y_init, dim_x,dim_y)
        for bbox in bboxes:
            # Calculate the scaling factors for width and height
            scale_x = dim_x / dim_x_init
            scale_y = dim_y / dim_y_init
            print(scale_x,scale_y)
            # Convert the coordinates to the new dimensions
            bbox_resized = [ int(bbox[0] * scale_x), int(bbox[1] * scale_y), int(bbox[2] * scale_x), int(bbox[3] * scale_y)]
            bboxes_resized.append(bbox_resized)
        print(bboxes_resized)

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

    transform = transforms.Compose([
        ai8x.normalize(args=args)
    ])

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=30, translate=(0.5, 0.5), scale=(0.5,1.5), fill=0),
            
            transforms.Resize((160,120)),
            transforms.ToTensor(),
            ai8x.normalize()
        ])

        train_dataset = WIDERFacesDataset(data_dir=os.path.join(data_dir, "WIDER_faces", "train"), transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160,120)),
            transforms.ToTensor(),
            ai8x.normalize()
        ])
        # Load validation dataset instead of test dataset, since test dataset is unlabeled
        test_dataset = WIDERFacesDataset(data_dir=os.path.join(data_dir, "WIDER_faces", "val"), transform=train_transform)
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'widerfaces',
        'input': (3, 160, 120),
        'output': [('x', float), ('y', float), ('w', float), ('h', float)],
        'regression': True,
        'loader': widerfaces_get_datasets,
    },
]