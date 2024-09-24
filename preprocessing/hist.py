import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as albu
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pretrained_microscopy_models as pmm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_dir = '/vol/bitbucket/zy3023/code/alloyed/dimi/pipeline/' # 'directory path containing the inference scripts and everything, basically the directory path of this script'
output_dir = Path(base_dir, 'output')# 'subdirectory in which the generated masks and visualizations will be saved')
os.makedirs(output_dir, exist_ok=True)
input_dir = Path(base_dir, 'original_input')
equal_input_dir = Path(base_dir, 'equalized_input')

visualizations_dir = output_dir / 'visualizations'
carbide_mask_dir = output_dir / 'carbide_masks'
gamma_prime_mask_dir = output_dir / 'gamma_prime_masks'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)
os.makedirs(carbide_mask_dir, exist_ok=True)
os.makedirs(gamma_prime_mask_dir, exist_ok=True)
os.makedirs(equal_input_dir, exist_ok=True)

for filename in tqdm(os.listdir(input_dir)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = input_dir / filename
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to YUV color space
        yuv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)

        # Equalize the histogram of the Y channel
        yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])

        # Convert the image back to RGB color space
        equalized_image_rgb = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)

        # Convert the equalized RGB image back to BGR
        equalized_image_bgr = cv2.cvtColor(equalized_image_rgb, cv2.COLOR_RGB2BGR)

        # Save the image in BGR format
        cv2.imwrite(str(equal_input_dir / f"{filename}.png"), equalized_image_bgr)#.astype(np.uint8))
