# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class TorchImageProcessor:
    """Simple data processors"""

    def __init__(self, 
                 image_size, 
                 is_color, 
                 mean, 
                 scale,
                 crop_size, 
                 pad=28, 
                 color='BGR',
                 use_cutout=False,
                 use_mirroring=False,
                 use_random_crop=False,
                 use_center_crop=False,
                 use_random_gray=False):
        """Everything that we need to init"""
        trs = []
        trs.append(transforms.RandomRotation(20, expand= True))
        if use_mirroring:
            #trs.append(transforms.RandomVerticalFlip(p=1))
            trs.append(transforms.RandomHorizontalFlip())
        if use_random_gray:
            trs.append(transforms.RandomGrayscale)
        if use_random_crop:
            trs.append(transforms.Pad(pad))
            trs.append(transforms.RandomCrop(crop_size))
        elif use_center_crop:
            trs.append(transforms.Pad(pad))
            trs.append(transforms.CenterCrop(crop_size))
        trs.append(transforms.Resize(image_size))
        trs.append(transforms.ToTensor())
        self.transform = transforms.Compose(trs)


    def process(self, image_path):
        """
        Returns processed data.
        """

        image = Image.open(image_path.replace('\\', '/'))


        if image is None:
            print(image_path)

        # TODO: .......... ......... ........... ........... ......... OpenCV . TorchVision
        # .. ...... ....... ......... ...... numpy . .............. ......... ........

        return self.transform(image).numpy()