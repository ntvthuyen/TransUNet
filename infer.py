import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
#from datasets.dataset_eyetracking import Eyetracking_dataset
#from datasets.dataset_synapse import Synapse_dataset
from utils import infer
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchvision import transforms
from imgaug import augmenters as iaa

class TransUNetSegmentation:
    def __init__(self):
        
        self.deterministic = True
        self.dataset = "OpenEDS"
        self.n_skip = 3
        self.vit_name = "R50-ViT-B_16"
        self.seed = 1234 
        self.vit_patches_size = 16
        self.img_size = 224
        if not self.deterministic:
            cudnn.benchmark = True
            cudnn.deterministic = False
        else:
            cudnn.benchmark = False
            cudnn.deterministic = True
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.num_classes = 9
        self.z_spacing = 1#dataset_config[dataset_name]['z_spacing']
        self.is_pretrain = True
        
    # modified


    # name the same snapshot defined in train script!
        self.exp = 'TU_pretrain_R50-ViT-B_16_skip3_10k_epo20_bs24_224' #'TU_pretrain_R50-ViT-B_16_skip3_10k_epo15_bs24_224' #'TU_' + dataset_name + str(args.img_size)
        snapshot_path = "TransUnetES/model/TU_OpenEDS224/" + self.exp
         
        config_vit = CONFIGS_ViT_seg[self.vit_name]
        config_vit.n_classes = self.num_classes
        config_vit.n_skip = self.n_skip
        config_vit.patches.size = (self.vit_patches_size, self.vit_patches_size)
        if self.vit_name.find('R50') !=-1:
            config_vit.patches.grid = (int(self.img_size/self.vit_patches_size), int(self.img_size/self.vit_patches_size))
        net = ViT_seg(config_vit, img_size=self.img_size, num_classes=config_vit.n_classes).cuda()

        snapshot = os.path.join(snapshot_path, 'best_model.pth')
        if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_19')
        net.load_state_dict(torch.load(snapshot))
 
        self.model = net 
        self.model.eval()
        self.resize_transform = transforms.Compose([
                np.asarray,
                    iaa.Sequential([
                    iaa.Resize({"height": self.img_size, "width": self.img_size})
                    ]).augment_image,
                np.copy,
                transforms.ToTensor(),
                ]
          ) 
 
    def infer(self, img, test_save_path=None):
        input = self.resize_transform(img)
        img_itk, prd_itk = infer(input, self.model, classes=self.num_classes, patch_size=[self.img_size, self.img_size], z_spacing=self.z_spacing)
        return prd_itk
