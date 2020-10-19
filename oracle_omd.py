import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import pandas as pd
import os
from oc_data_load import CIFAR10_Data

def known_unknown_split(split=0, normalize=False):
    ''' 
    Known/Unknown split semantics can be found in download_cifar10.py
    in the following git repo:
    https://github.com/lwneal/counterfactual-open-set

    I use a modified download script to produce csv files instead of
    JSON. Refer to download_cifar10_to_csv.py for further details

    NOTE: There are 5 possible splits that can be used here,
          the default split number is 0 unless specified otherwise.
    '''
    if normalize:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.247 , 0.243 , 0.261))
        ])
    else:
        # Just this by default
        transform = T.ToTensor()
        
    known_train   = CIFAR10_Data(csv_file='data/cifar10-split{}a.dataset'.format(split),
                                root_dir='data/cifar10', fold='train',
                                transform=transform)
    known_test    = CIFAR10_Data(csv_file='data/cifar10-split{}a.dataset'.format(split),
                                root_dir='data/cifar10', fold='test',
                                transform=transform)
    unknown_train = CIFAR10_Data(csv_file='data/cifar10-split{}b.dataset'.format(split),
                                root_dir='data/cifar10', fold='train',
                                transform=transform)
    unknown_test  = CIFAR10_Data(csv_file='data/cifar10-split{}b.dataset'.format(split),
                                root_dir='data/cifar10', fold='test',
                                transform=transform)
    
    return known_train, known_test, unknown_train, unknown_test


def omd():
    '''
    Training a linear anomaly detector
    '''
    pass


def kfold():
    pass


def get_weight_prior():
    '''TODO: Either use LODA or Isolation Forest'''
    pass


def main():
    # Load known and unknown classes (*a is known *b is unknown)
    train_known, test_known, train_unknown, test_unknown = known_unknown_split()
    # binary or multiclass category detector??

if __name__ == '__main__':
    main()
