import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd

class oc_cifar10_data_loader():
    def __init__(self):
        self.classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog',
            'frog', 'horse', 'ship', 'truck')
        self.default_unknown = ['airplane', 'cat', 'frog', 'ship']
        # Can download CIFAR-10 at the url below
        # https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
        
