import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import pandas as pd
import os
from oc_data_load import CIFAR10_Data
from vanilla_ae import get_vanilla_ae

def load_data(split=0, normalize=False):
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
        
    kn_train   = CIFAR10_Data(csv_file='data/cifar10-split{}a.dataset'.format(split),
                                root_dir='data/cifar10', fold='train',
                                transform=transform)
    kn_val    = CIFAR10_Data(csv_file='data/cifar10-split{}a.dataset'.format(split),
                                root_dir='data/cifar10', fold='val',
                                transform=transform)
    kn_test    = CIFAR10_Data(csv_file='data/cifar10-split{}a.dataset'.format(split),
                                root_dir='data/cifar10', fold='test',
                                transform=transform)
    
    unkn_train = CIFAR10_Data(csv_file='data/cifar10-split{}b.dataset'.format(split),
                                root_dir='data/cifar10', fold='train',
                                transform=transform)
    unkn_val  = CIFAR10_Data(csv_file='data/cifar10-split{}b.dataset'.format(split),
                                root_dir='data/cifar10', fold='val',
                                transform=transform)
    unkn_test  = CIFAR10_Data(csv_file='data/cifar10-split{}b.dataset'.format(split),
                                root_dir='data/cifar10', fold='test',
                                transform=transform)

    
    return (kn_train, kn_val, kn_test, unkn_train, unkn_val, unkn_test)


def omd():
    '''
    Training a linear anomaly detector
    '''
    pass


def train_oracle_latent_rep():
    pass


def get_known_ae(kn_train, kn_val, filename='vanilla_ae.pth'):
    CIFAR10_DIM = 32*32
    NUM_EPOCHS = 20

    device = torch.device("cuda")
    model = get_vanilla_ae(kn_train, kn_val, filename)
    print(model.state_dict())


def get_weight_prior():
    '''TODO: Either use LODA or Isolation Forest'''
    pass


def main():
    # Load known and unknown classes
    kn_train, kn_val, kn_test, unkn_train, unkn_val, unkn_test = load_data()
    
    # binary or multiclass category detector??
    get_known_ae(kn_train, kn_val)
    
if __name__ == '__main__':
    main()
