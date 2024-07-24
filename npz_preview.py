import os
import numpy as np 
import cv2 
npz_path = "./data/Synapse/train_npz/"

npz_files = os.listdir(npz_path)

for npz_file_path in npz_files:
    
    
    npz_file = np.load(os.path.join(npz_path,npz_file_path))
    if np.sum(npz_file["label"]) == 0:
        continue
    cv2.imwrite("label.png", npz_file["label"]*100)
    cv2.imwrite("image.png", npz_file["image"]*100)

