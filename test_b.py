import os
import torch
import cv2
import random
import imageio
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import matplotlib.pyplot as plt
import pretrained_microscopy_models as pmm
import segmentation_models_pytorch as smp
import albumentations as albu

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from skimage import img_as_ubyte
from tqdm import tqdm
import argparse


def stitch_images(crops, positions, image_size, channel=3):
    if channel > 1:
        stitched_image = np.zeros((image_size[0], image_size[1], channel), dtype=np.uint8)
    else:
        stitched_image = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)

    for crop, (x, y) in zip(crops, positions):
        crop_height, crop_width = crop.shape[:2]
        y_end = min(y + crop_height, stitched_image.shape[0])
        x_end = min(x + crop_width, stitched_image.shape[1])
        stitched_image[y:y_end, x:x_end] = crop[:y_end - y, :x_end - x]
    return stitched_image

def sliding_window_crop(image_path, crop_size=(512, 512), step_size=256):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    crops, positions = [], []
    for y in range(0, height - crop_size[1] + 1, step_size):
        for x in range(0, width - crop_size[0] + 1, step_size):
            cropped_image = image[y:y + crop_size[1], x:x + crop_size[0]]
            crops.append(cropped_image)
            positions.append((x, y))
    return crops, positions, image.shape[:2]

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    # This is turned off for this dataset
    test_transform = [
        #albu.Resize(height,width)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def test_from_folder(path):

    ''' option 1. test from original folder '''
    for bb in tqdm(os.listdir(path)):
        im_path = f'{path}/{bb}'

        name = os.path.basename(im_path)

        # if you want to vis gt_mask, ste correct path here.
        img_list, positions, img_size = sliding_window_crop(im_path, crop_size=(args.crop_size, args.crop_size), step_size=args.crop_size)

        pred_list = []
        mask_list = []
        for image in img_list:
            mask = np.zeros_like(image) # if you want to vis gt_mask, ste correct path here.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[0] != image.shape[1]:
                raise ValueError

            assert image.shape[0] == image.shape[1]
            image = cv2.resize(image, (512,512)).astype(np.uint8)
            mask = cv2.resize(mask, (512,512)).astype(np.uint8)

            image_vis = image.copy()

            # extract certain classes from mask (e.g. cars)
            masks = [np.all(mask == v, axis=-1) for v in class_values.values()]
            if len(masks) > 1:
                masks[0] = ~np.any(masks[1:], axis=0)
            gt_mask = np.stack(masks, axis=-1).astype('float')
            gt_mask_tert = gt_mask.squeeze().round().astype('bool')

            sample = get_preprocessing(preprocessing_fn)(image=image, mask=gt_mask)
            image, gt_mask = sample['image'], sample['mask']

            x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
            pr_mask = best_model.predict(x_tensor)

            pr_mask_tert = pr_mask.squeeze().cpu().numpy().round().astype('bool')
            x = image_vis.copy()

            # carbide
            x[pr_mask_tert, 0] = x[pr_mask_tert, 0] * 0.5
            x[pr_mask_tert, 1] = x[pr_mask_tert, 0] * 0.5 + 255 * 0.5
            x[pr_mask_tert, 2] = x[pr_mask_tert, 2] * 0.5

            # GT
            # gt = image_vis.copy()
            # gt[gt_mask_tert, 0] = gt[gt_mask_tert, 0] * 0.5
            # gt[gt_mask_tert, 1] = 255
            # gt[gt_mask_tert, 2] = gt[gt_mask_tert, 2] * 0.5
            pred_list.append(x)
            mask_list.append(pr_mask_tert)

        image_vis = stitch_images(img_list, positions, img_size)
        x = stitch_images(pred_list, positions, img_size)
        pr_mask_tert = stitch_images(mask_list, positions, img_size, channel=1)

        p = {}
        p['image'] = image_vis
        # p['ground_truth_mask_overlay'] = gt
        p['predicted_mask_overlay'] = x
        p['mask'] = pr_mask_tert
        pmm.util.visualize(
            p, name, out_path
        )


if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description='Fitting'
    )
    parser.add_argument('--path', '-p', default='./example_data_ti/train/', type=str, help='Path to image & mask folder.')
    parser.add_argument('--class_name', '-c', default='primary', type=str, help='class name of the segmented class.')
    parser.add_argument('--ckpt_path', default='/data/home/zy3023/alloyed/Ti_alloys/models_ti/', type=str, help='Path to checkpoints.')
    parser.add_argument('--crop_size', default=512, type=int, help='image patch size')
    args = parser.parse_args()

    out_path = f'./output_{args.class_name}/'

    ####################
    # model parameters
    architecture = 'UnetPlusPlus'
    encoder = 'resnet50'
    pretrained_weights = 'micronet'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # how the images will be normalized. Use imagenet statistics even on micronet pre-training
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet') 
    # test_img_dir = os.path.join(args.path, 'train')
    # test_annot_dir = os.path.join(args.path, f'train_annot_{args.class_name}')

    # pixel values of the annotations for each mask.
    # class_values = {'matrix': [0,0,0],
    #                'secondary': [255,0,0],
    #                'tertiary' : [0,0,255]}
    # class_values = {'background': [0],
    # 'oxide': [1],
    # 'cracks' : [2]}
    class_values = {args.class_name: [0]}

    # load best model
    best_model_path = Path(args.ckpt_path, args.class_name, 'model_best.pth.tar')
    state = torch.load(best_model_path) 
    best_model = pmm.segmentation_training.create_segmentation_model(
        architecture=architecture,
        encoder=encoder,
        encoder_weights=pretrained_weights,
        classes=1 # secondary precipitates, tertiary precipitates, matrix
        )
    best_model.load_state_dict(pmm.util.remove_module_from_state_dict(state['state_dict']))
    best_model = best_model.cuda()

    test_from_folder(args.path)
