import mmpose
import random
from numpy.lib import isin
import torch
import cv2

import numpy as np
import torchvision.transforms as transforms

from mmpose.datasets import PIPELINES
from PIL import Image

@PIPELINES.register_module()
class ColorJitter:
    """Data augmentation with random Brightness, Contrast and Saturation.

    Required keys: 'img'
    Modifies key: 'img'

    Args:
        brightness (float): 0-1 
        contrast (float): 0-1 
        saturation (float): 0-1 
        hue (float): 0-1 
    """

    def __init__(self, probability=0.5, brightness=0.5, contrast=0.5, saturation=0.05, hue=0.05):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation= saturation
        self.hue = hue
        self.probability = probability
    
    def __call__(self, results):
       img = results['img']
       pil_img = Image.fromarray(img)
       if random.randint(1,100)< self.probability*100:
           transform = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation)
           transformed = transform(pil_img)
           results['img'] = np.array(transformed)
       return results



@PIPELINES.register_module()
class RandomGrayScale:
    """Data augmentation with random Brightness, Contrast and Saturation.

    Required keys: 'img'
    Modifies key: 'img'

    Args:
        probability (float): 0-1 random gray convert probability
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, results):
       img = results['img']
       pil_img = Image.fromarray(img)
       transform = transforms.RandomGrayscale(self.probability)
       transformed = transform(pil_img)
       results['img'] = np.array(transformed)
       return results




@PIPELINES.register_module()
class RandomErase:
    """Data augmentation with random erase.

    Required keys: 'img'
    Modifies key: 'img'

    Args:
        probability (float): 0-1 random gray convert probability
    """

    def __init__(self, probability=0.5, scale=(0.01,0.01)):
        self.probability = probability
        self.scale = scale

    def __call__(self, results):
        # TODO: Probability implementation
       img = results['img']
       pil_img = Image.fromarray(img)
       transform = transforms.Compose([transforms.ToTensor(),transforms.RandomErasing(scale=[0.01,0.01]),transforms.ToPILImage()])
       transformed = transform(pil_img)
       results['img'] = np.array(transformed)
       return results

@PIPELINES.register_module()
class RandomHE:
    """Data augmentation with random erase.

    Required keys: 'img'
    Modifies key: 'img'

    Args:
        probability (float): 0-1 random gray convert probability
        type: Valid types- ['simple' (cv2.equalizeHist), 'CLAHE' (cv2.createClahe), 'all' (Random choice from all) ]
        options: 
            CLAHE: ['clipLimit', 'tileGridSize']
            type: [int, 2-tuple]
            deafult: [4, (10,10)]
    """

    def __init__(self, probability=0.5, type='clahe', options=None):
        self.probability = probability

        self.equalizers = {
            'simple': self.equalize_hist_simple,
            'clahe': self.equalize_hist_clahe,
            'all': self.equalize_hist_all
            }

        self.types_HE = list(self.equalizers)

        if type not in self.types_HE:
            raise KeyError(f"Invalid Histogram Equalization type input. Type can be among: {self.types_HE}")

        self.type = type

        VALID_OPTS = {'clahe': ['clipLimit', 'tileGridSize']}
        self.opts = options
    
    def equalize_hist_simple(self, gray_img):
        return cv2.equalizeHist(gray_img)
    

    def equalize_hist_clahe(self, gray_img):
        
        clipLimit = self.opts['clipLimit'] if 'clipLimit' in list(self.opts['clahe']) else 4            # For equalize_hist_all, these default values will be taken, if CLAHE opts not provided
        tileGridSize = self.opts['tileGridSize'] if 'tileGridSize' in list(self.opts['clahe']) else (10,10) # For equalize_hist_all, these default values will be taken, if CLAHE opts not provided

        if not isinstance(clipLimit, int):
            raise TypeError(f"clipLimit argument for CLAHE should be of type 'int'. Current input type: {type(clipLimit)}")
        
        if not isinstance(tileGridSize, int):
            raise TypeError(f"tileGridSize argument for CLAHE should be of type '2-tuple'. Current input type: {type(tileGridSize)}")

        clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=tileGridSize)
        equalized = clahe.apply(gray_img)

        return equalized
    

    def equalize_hist_all(self, gray_img):
        equalizers= list(self.equalizers)[:-1]
        select_eq = equalizers[random.randint(0,len(equalizers)-1)]
        inputs = {'gray_img': gray_img}
        equalized = self.equalizers[select_eq](**inputs)

        return equalized


    def __call__(self, results):
        img = results['img']
        do_HE =  True if (random.randint(0,(1/self.probability)-1)) ==0 else False
        if do_HE:
            if img.shape[2] == 1:
                gray = img
            elif img.shape[2] ==3:
                gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) #Assume 3 channel = RGB
            else:
                raise AssertionError("Input image channels is not 1 or 3")

            inputs = {'gray_img': gray}
            results['img'] = self.equalizers[self.type](**inputs)
        return results
