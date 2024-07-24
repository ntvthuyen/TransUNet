import nibabel as nib 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import os

save_path = './prediction_viz/'
nii_path = "./predictions/TU_pretrain_R50-ViT-B_16_skip3_10k_epo15_bs24_224/TU_pretrain_R50-ViT-B_16_skip3_10k_epo15_bs24_224/"

def convert_gray2rgb(image):
    width, height = image.shape
    out = np.empty((width, height, 3), dtype=np.uint8)
    out[:, :, 0] = image
    out[:, :, 1] = image
    out[:, :, 2] = image
    return out

def draw_mask(image, mask_generated) :
    image = convert_gray2rgb(image)
    
    masked_image = image.copy()
   
    mask_generated = convert_gray2rgb(mask_generated)  #cv2.cvtColor(mask_generated, cv2.COLOR_GRAY2RGB) 
    masked_image = np.where(mask_generated.astype(int)==1,
                          np.array([0,255,0], dtype='uint8'),
                          masked_image)
    
    masked_image = np.where(mask_generated.astype(int)==2,
                          np.array([0,0,255], dtype='uint8'),
                          masked_image)


    masked_image = masked_image.astype(np.uint8)

    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)

def visualize(nii_path, image_name, save_path="./predictions_viz"):
    if "img.nii" not in image_name:
        return
    
    image_path = os.path.join(nii_path, image_name)

    nii_img = nib.load(image_path)
    nii_img_data = nii_img.get_fdata()
    
    image_path = image_path.replace("img.nii", "pred.nii")
    nii_pred = nib.load(image_path) 
    nii_pred_data = nii_pred.get_fdata()
    
    print(nii_pred_data)
    output = draw_mask(nii_img_data, nii_pred_data)

    cv2.imwrite(os.path.join(save_path, image_name.replace(".nii.gz",".jpg")), output)


if __name__ == "__main__":
    nii_files = os.listdir(nii_path)

    for i in nii_files:
        visualize(nii_path, i)
