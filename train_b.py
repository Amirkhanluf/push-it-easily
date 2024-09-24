import os
import torch
import cv2
import random
import imageio
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import matplotlib.pyplot as plt
import pretrained_microscopy_models as pmm
import segmentation_models_pytorch as smp
import albumentations as albu
import argparse

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

# set random seeds for repeatability
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def get_training_augmentation():
    train_transform = [
        albu.Flip(p=0.75),
        albu.RandomRotate90(p=1),       
        albu.GaussNoise(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1, limit=0.25),
                albu.RandomGamma(p=1),
            ],
            p=0.50,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                #albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.50,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1, limit=0.3),
                albu.HueSaturationValue(p=1),
            ],
            p=0.50,
        ),
    ]
    return albu.Compose(train_transform)

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


if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description='Fitting'
    )
    parser.add_argument('--path', '-p', default='./example_data_ti/', type=str, help='Path to image & mask folder.')
    parser.add_argument('--class_name', '-c', default='primary', type=str, help='class name of the segmented class.')
    parser.add_argument('--ckpt_path', default='/data/home/zy3023/alloyed/checkpoints/', type=str, help='Path to checkpoints.')
    parser.add_argument('--crop_size', default=512, type=int, help='Image patch size')
    parser.add_argument('--step_size', default=256, type=int, help='Sliding windown size used in training')
    args = parser.parse_args()

    # model parameters
    architecture = 'UnetPlusPlus'
    encoder = 'resnet50'
    pretrained_weights = 'micronet'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the Unet model with a resnet backbone that is pre-trained on micronet
    print('creating model...')
    model = pmm.segmentation_training.create_segmentation_model(
        architecture=architecture,
        encoder = encoder,
        encoder_weights=pretrained_weights,
        classes=1 # secondary precipitates, tertiary precipitates, matrix
        )

    x_train_dir = os.path.join(args.path, 'train')
    y_train_dir = os.path.join(args.path, f'train_annot_{args.class_name}')

    x_valid_dir = os.path.join(args.path, 'train')
    y_valid_dir = os.path.join(args.path, f'train_annot_{args.class_name}')

    # x_test_dir = os.path.join(args.path, 'train')
    # y_test_dir = os.path.join(args.path, f'train_annot_{args.class_name}')

    # how the images will be normalized. Use imagenet statistics even on micronet pre-training
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet') 

    # pixel values of the annotations for each mask.
    class_values = {args.class_name: [0]} # for Ti alloyed images, here is 0, otherwise it should be 1

    training_dataset = pmm.io.Dataset(
        images=x_train_dir,
        masks=y_train_dir,
        class_values=class_values,
        is_train=True,
        crop_size=args.crop_size,
        step_size=args.step_size,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    validation_dataset = pmm.io.Dataset(
        images=x_valid_dir,
        masks=y_valid_dir,
        class_values=class_values,
        is_train=True,
        crop_size=args.crop_size,
        step_size=args.step_size,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    # test_dataset = pmm.io.Dataset(
    #     images=x_test_dir,
    #     masks=y_test_dir,
    #     class_values=class_values,
    #     augmentation=get_validation_augmentation(),
    #     preprocessing=get_preprocessing(preprocessing_fn)
    # )

    #########################################################
    # DEBUG
    # validation data
    # visualize_dataset = pmm.io.Dataset(
    #     images=x_valid_dir,
    #     masks=y_valid_dir,
    #     class_values=class_values,
    #     augmentation=get_validation_augmentation(),
    #     #preprocessing=get_preprocessing(preprocessing_fn)
    # )

    # v = {}
    # for im, mask, name in visualize_dataset:
    #     v['image'] = im
    #     v['oxide_mask'] = mask.squeeze()
    #     pmm.util.visualize(
    #         v,
    #         name
    #     )
    #########################################################

    print('start training, dataset length: ', len(training_dataset))
    state = pmm.segmentation_training.train_segmentation_model(
        model=model,
        architecture=architecture,
        encoder=encoder,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        class_values=class_values,
        epochs=80,
        patience=30,
        device=device,
        lr=2e-4,
        batch_size=6,
        val_batch_size=6,
        save_folder=os.path.join(args.ckpt_path, args.class_name),
        save_name='binary_segmentation_example.pth.tar'
    )

    plt.plot(state['train_loss'], label='train_loss')
    plt.plot(state['valid_loss'], label='valid_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss.jpg')

    # drop the learning rate and keep training to see if we can squeeze a little more out.
    model_path = Path(args.ckpt_path, args.class_name, 'binary_segmentation_example.pth.tar')
    state = pmm.segmentation_training.train_segmentation_model(
        model=str(model_path),
        architecture=architecture,
        encoder=encoder,
        epochs=120,
        patience=30,
        device=device,
        lr=1e-5,
        batch_size=6,
        val_batch_size=6,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        class_values=class_values,
        save_folder=os.path.join(args.ckpt_path, args.class_name),
        save_name='binary_segmentation_example_low_lr.pth.tar'
    )
