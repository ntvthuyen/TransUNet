import os
import random
import h5py
import numpy as np
import torch
import glob
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
from torchvision import transforms
IMAGE_SIZE = 224

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Eyetracking_dataset(Dataset):
    def __init__(self, 
                 openEDS_data_path="/mnt/f/sianus/openEDS/openEDS/S_*/*.png", 
                 manual_label_data_path="/mnt/f/sianus/openEDS/openEDS/manual_labeled_data/*.npz", 
                 split="train", 
                 transform=None):
        local_random = random.Random(676)
        self.resize_transform = transforms.Compose([
                np.asarray,
                    iaa.Sequential([
                    iaa.Resize({"height": IMAGE_SIZE, "width": IMAGE_SIZE})
                    ]).augment_image,
                np.copy,
                #transforms.ToTensor(),
                #transforms.Normalize(
                    #mean = [0.485, 0.456, 0.406],
                    #std = [0.229, 0.224, 0.225])
                ]
          )

        self.transform = transform  # using transform in torch!
        self.split = split
        self.openEDS_sample_list = glob.glob(openEDS_data_path)
        self.manual_label_sample_list = glob.glob(manual_label_data_path)
        local_random.shuffle(self.openEDS_sample_list)
        local_random.shuffle(self.manual_label_sample_list)
        #self.openEDS_sample_list_length = len(self.openEDS_sample_list)
        #self.manual_label_sample_list_length = len(self.openEDS_sample_list_length)
        #self.sample_list = #open(os.path.join(list_dir, self.split+'.txt')).readlines()
        #self.data_dir = base_dir
        if self.split == "train":
            #print(self.openEDS_sample_list)
            self.openEDS_length = int(len(self.openEDS_sample_list)*0.7)
            self.manual_length = int(len(self.manual_label_sample_list)*0.7)
            self.openEDS_sample_list = self.openEDS_sample_list[:self.openEDS_length]
            self.manual_label_sample_list = self.manual_label_sample_list[:self.manual_length]
        elif self.split == "test_manual":
            #self.openEDS_length = 0 #int(len(self.openEDS_sample_list)*0.7)
            self.manual_length = int(len(self.manual_label_sample_list)*0.7)
            # self.openEDS_sample_list = self.openEDS_sample_list[self.openEDS_length:]
            self.manual_label_sample_list = self.manual_label_sample_list[self.manual_length:]
            self.openEDS_length = 0#len(self.openEDS_sample_list) #- int(len(self.openEDS_sample_list)*0.7)
            self.manual_length = len(self.manual_label_sample_list) #- int(len(self.manual_label_sample_list)*0.7)
         
 
            #return self.openEDS_length + self.manual_length
        else:
            self.openEDS_length = int(len(self.openEDS_sample_list)*0.7)
            self.manual_length = int(len(self.manual_label_sample_list)*0.7)
            self.openEDS_sample_list = self.openEDS_sample_list[self.openEDS_length:]
            self.manual_label_sample_list = self.manual_label_sample_list[self.manual_length:]
            self.openEDS_length = len(self.openEDS_sample_list) #- int(len(self.openEDS_sample_list)*0.7)
            self.manual_length = len(self.manual_label_sample_list) #- int(len(self.manual_label_sample_list)*0.7)
         
    def __len__(self):
        return self.openEDS_length + self.manual_length

    def __getitem__(self, idx):
        #if self.split == "train":
        if idx < self.openEDS_length:
            data_path = self.openEDS_sample_list[idx][:-3] + "npy"
            label = np.load(data_path)
            _idx1 = label == 1
            _idx2 = label == 2 
            _idx3 = label == 3
            label[_idx1] = 0
            label[_idx2] = 1 
            label[_idx3] = 2
            image = cv2.imread(data_path[:-3]+"png", cv2.IMREAD_GRAYSCALE)
        else:
            data_path = self.manual_label_sample_list[idx - self.openEDS_length]
            label = np.load(data_path)
            label = label["arr_0"]
            _idx1 = label==1
            _idx2 = label==2
            label[_idx1] = 2 
            label[_idx2] = 1 
            image = cv2.imread(data_path[:-3] + "png", cv2.IMREAD_GRAYSCALE)
        #print(image.shape, label.shape) 
            #else:
            #vol_name = self.sample_list[idx].strip('\n')
            #filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            #data = h5py.File(filepath)
            #image, label = data['image'][:], data['label'][:]
        image = self.resize_transform(image)
        label = self.resize_transform(label)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = data_path[23:].replace("/","") #self.sample_list[idx].strip('\n')
        return sample
