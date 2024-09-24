import cv2
import os
import numpy as np
import random
from torch.utils.data import Dataset as BaseDataset
from PIL import Image

def stitch_images(crops, positions, image_size):
    stitched_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    for crop, (x, y) in zip(crops, positions):
        crop_height, crop_width = crop.shape[:2]
        y_end = min(y + crop_height, stitched_image.shape[0])
        x_end = min(x + crop_width, stitched_image.shape[1])
        stitched_image[y:y_end, x:x_end] = crop[:y_end - y, :x_end - x]
    return stitched_image

def sliding_window_crop(image_path, crop_size, step_size):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    crops, positions = [], []
    for y in range(0, height - crop_size[1] + 1, step_size):
        for x in range(0, width - crop_size[0] + 1, step_size):
            cropped_image = image[y:y + crop_size[1], x:x + crop_size[0]]
            crops.append(cropped_image)
            positions.append((x, y))
    return crops

class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    Modified from https://github.com/qubvel/segmentation_models.pytorch
    
    
    Args:
        images (str or list): path to images folder or list of images
        masks (str): path to segmentation masks folder or list of images
        class_values (dict): values of classes to extract from segmentation mask. 
            Each dictionary value can be an integer or list that specifies the mask
            values that belong to the class specified by the corresponding dictionary key.
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    Note: If images and masks are directories the image and mask pairs should be 
    laballed "ImageName.tif" and "ImageNamemask.tif" respectively. Otherwise
    you should just pass the list of paths to images and masks.
    """
    
    def __init__(
            self, 
            images, 
            masks, 
            class_values,
            is_train,
            crop_size,
            step_size,
            augmentation=None, 
            preprocessing=None,
    ):
        self.is_train = is_train
        self.class_values = class_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # create list of image paths
        if type(images) is list:
            self.images_fps = images
            self.masks_fps = masks
        else:
            self.ids = os.listdir(images)
            self.images_fps = [os.path.join(images, image_id) for image_id in self.ids]
            self.masks_fps = [os.path.join(masks, image_id) for image_id in self.ids]

        # Eliminate samples without GT mask.
        i_list = []
        j_list = []
        for i, j in zip(self.images_fps, self.masks_fps):
            if not os.path.exists(j):
                i_list.append(i)
                j_list.append(j)

        for i, j in zip(i_list, j_list):
            self.images_fps.remove(i)
            self.masks_fps.remove(j)

        # Crop image into smaller pieces.
        original_image = cv2.imread(self.images_fps[0])
        self.img_list = []
        self.mask_list = []
        if self.is_train:
            for img_name, mask_name in zip(self.images_fps, self.masks_fps):
                x = sliding_window_crop(img_name, (crop_size, crop_size), step_size)
                y = sliding_window_crop(mask_name, (crop_size, crop_size), step_size)
                self.img_list += x
                self.mask_list += y
        else:
            raise NotImplementedError
            for img_name, mask_name in zip(self.images_fps, self.masks_fps):
                x = sliding_window_crop(img_name, (crop_size, crop_size), crop_size)
                y = sliding_window_crop(mask_name, (crop_size, crop_size), crop_size)
                self.img_list += x
                self.mask_list += y

        self.n_patch = len(self.img_list)
        print(f'Trianing on {self.n_patch} patches..., crop_size {crop_size}, step_size {step_size}')

    def __len__(self):
        return self.n_patch

    def __getitem__(self, i):

        image = self.img_list[i] # NOTE: make sure here is BGR
        mask = self.mask_list[i]

        assert image.shape[0] == image.shape[1]
        # image = cv2.resize(image, (512,512)).astype(np.uint8)
        # mask = cv2.resize(mask, (512,512)).astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # extract certain classes from mask (e.g. cars)
        masks = [np.all(mask == v, axis=-1) for v in self.class_values.values()]
        if len(masks) > 1:
            masks[0] = ~np.any(masks[1:], axis=0)
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.is_train:
            return image, mask
        else:
            return image, mask, self.ids[i]

    def resample(self):
        return self.__getitem__(random.randint(0,len(self)))
