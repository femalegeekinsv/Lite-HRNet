import mmpose
import random
import torch
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
       img = results['img']
       pil_img = Image.fromarray(img)
       transform = transforms.Compose([transforms.ToTensor(),transforms.RandomErasing(scale=[0.01,0.01]),transforms.ToPILImage()])
       transformed = transform(pil_img)
       results['img'] = np.array(transformed)
       return results