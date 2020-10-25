import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import pandas as pd
from pyod.models.loda import LODA
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
        
    kn_train   = CIFAR10_Data(root_dir='data/cifar10',
                              csv_file='data/cifar10-split{}a.dataset'.format(split),
                              fold='train',
                              transform=transform)
    kn_val    = CIFAR10_Data(root_dir='data/cifar10',
                             csv_file='data/cifar10-split{}a.dataset'.format(split),
                             fold='val',
                             transform=transform)
    kn_test    = CIFAR10_Data(root_dir='data/cifar10',
                              csv_file='data/cifar10-split{}a.dataset'.format(split),
                              fold='test',
                              transform=transform)
    
    unkn_train = CIFAR10_Data(root_dir='data/cifar10',
                              csv_file='data/cifar10-split{}b.dataset'.format(split),
                              fold='train',
                              transform=transform)
    unkn_val  = CIFAR10_Data(root_dir='data/cifar10',
                             csv_file='data/cifar10-split{}b.dataset'.format(split),
                             fold='val',
                             transform=transform)
    unkn_test  = CIFAR10_Data(root_dir='data/cifar10',
                              csv_file='data/cifar10-split{}b.dataset'.format(split),
                              fold='test',
                              transform=transform)

    #kn_train = torch.utils.data.DataLoader(kn_train, batch_size=4, pin_memory=True)
    #kn_val = torch.utils.data.DataLoader(kn_val, batch_size=4, pin_memory=True)
    #kn_test = torch.utils.data.DataLoader(kn_test, batch_size=4, pin_memory=True)
    #unkn_train = torch.utils.data.DataLoader(unkn_train, batch_size=4, pin_memory=True)
    #unkn_val = torch.utils.data.DataLoader(unkn_val, batch_size=4, pin_memory=True)
    #unkn_test = torch.utils.data.DataLoader(unkn_test, batch_size=4, pin_memory=True)

    return (kn_train, kn_val, kn_test, unkn_train, unkn_val, unkn_test)


def omd():
    '''
    Training a linear anomaly detector
    '''
    pass


def train_oracle_latent_rep():
    pass


def get_plain_ae(kn_train, kn_val, filename='plain_ae.pth'):
    CIFAR10_DIM = 32*32
    NUM_EPOCHS = 20
    
    #device = torch.device("cuda")
    model = get_vanilla_ae(kn_train, kn_val, filename)
    return model


def get_weight_prior(X_val_latent):
    '''
    Get weight vector prior using LODA (Pevny16)
    
    Parameters
    ----------
    X_val_latent : numpy array
        Describes the latent representation of images in the
        validation set. Note: This contains known and unknown
        examples.

    Returns
    -------
    learned weights associated with each latent feature. Used as the
    prior in training a linear anomaly detector on all classes
    given latent representation from a model trained on only 6.
    
    '''
    #contamination = 0.4 # 6 known classes, 4 unknown using CIFAR10
    #n_bin = len()
    clf_name = 'LODA'
    clf = LODA()
    clf.fit(X_val_latent)

    # y_val_latent_pred = clf.labels_ # binary (0: inlier, 1: outlier)
    # y_train_scores = clf.decision_scores_ # raw outlier scores

    return clf.get_params() # By default deep=True


def construct_column_labels(data_sample):
    '''Builds a list of labels that will be used to label
    columns in dataframe representing latent data'''
    num_features = len(data_sample)
    # This could be, for example, the size of the
    # latent space representation
    feature_list = []
    for i in range(num_features):
        feature_list.append('f_{}'.format(str(i)))
        
    feature_list.append('label')
    return feature_list


def concat_design_and_target(dataset): #, metadata):
    ''' 
    Embeds labels with training examples. 
    Utility for building latent representation dataset
    '''
    concat_data = []
    df = dataset.frame
    num_examples = len(dataset)
    for i in range(num_examples):
        concat_data.append([dataset[i], df.iloc[i]['label']])
         
    return concat_data


def construct_latent_set(model, kn_dataset, unkn_dataset):
    '''Build dataset from latent representation given
    a model that acts as the encoder and a dataset of
    raw data that is transformed by the encoder'''
    kn_X_y = concat_design_and_target(kn_dataset)#, metadata)
    unkn_X_y = concat_design_and_target(unkn_dataset)
    kn_unkn_X_y = torch.utils.data.ConcatDataset([kn_X_y, unkn_X_y])
    loader = torch.utils.data.DataLoader(kn_unkn_X_y, batch_size=1, shuffle=True)
    col_labels_loaded = False
    col_labels = []
    embed_list = []
    
    # NOTE: Each image batch consists of one image
    for i, (img_batch, label) in enumerate(loader):
        latent_batch = model.get_latent(img_batch)
        embedding = torch.reshape(torch.squeeze(latent_batch), (-1,))
        # TODO: Append this embedding to a pd dataframe with its label
        if col_labels_loaded == False:
            col_labels = construct_column_labels(embedding)
            val_latent_rep_df = pd.DataFrame(columns=col_labels)
            col_labels_loaded = True
            print("Populating Latent Dataframe")

        embedding = embedding.tolist()
        embedding.append(label[0])
        val_latent_rep_df.loc[i] = embedding

    print("Latent Dataframe Loading Complete")
    print(val_latent_rep_df)
    return val_latent_rep_df


def main():
    # Get datasets of known and unknown classes
    kn_train, kn_val, kn_test, unkn_train, unkn_val, unkn_test = load_data(0)
    
    # binary or multiclass category detector??
    kn_ae = get_plain_ae(kn_train, kn_val,'kn_std_ae_split_{}.pth'.format(0))

    # Training plain autoencoder on all training data
    kn_unkn_train = torch.utils.data.ConcatDataset([kn_train,unkn_train])
    # This preserves metadata
    # MIGHT NOT NEED kn_unkn_val_frame = pd.concat([kn_val.frame, unkn_val.frame])
    kn_unkn_val   = torch.utils.data.ConcatDataset([kn_val,  unkn_val  ])
    kn_unkn_ae = get_plain_ae(kn_unkn_train, kn_unkn_val,
                              'kn_unkn_std_ae_split_{}.pth'.format(0))

    # Get latent set used to train linear anomaly detector from the
    # validation set comprised of all classes
    #X, y = construct_latent_set(kn_ae, kn_unkn_val)
    latent_df = construct_latent_set(kn_ae, kn_val, unkn_val)

    # NEXT STEP: Use this latent data to train linear anomaly detector!! :)
    
    # build training set (X, y) for supervised latent classifier
    
    # Use latent space to train classifier AND as input to scoring function for
    # open category detector g
    
if __name__ == '__main__':
    main()
