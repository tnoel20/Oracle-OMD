import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import pandas as pd
import inspect
from pyod.models.loda import LODA
import pyod.utils.data as od
from pyod.utils.example import visualize

def get_synth_data(num_points=1000, num_features=128):
    
    data = np.random.randn(num_points, 3*num_features//4)
    data_anom = np.random.rand(num_points, num_features//4)
    data = np.hstack((data,data_anom))
    return pd.DataFrame.from_records(data)
    '''
    contamination = 0.5
    n_train = 200
    n_test = 100

    X_train, X_test, y_train, y_test = \
        od.generate_data(n_train=n_train, n_test=n_test, contamination=contamination, behaviour='new')
    print(X_train.shape)
    return X_train, X_test, y_train, y_test
    '''


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
    clf = LODA(n_bins=128)
    print(X_val_latent.shape)
    return clf.fit(X_val_latent)

    # y_val_latent_pred = clf.labels_ # binary (0: inlier, 1: outlier)
    # y_train_scores = clf.decision_scores_ # raw outlier scores

    #return clf.get_params() # By default deep=True


def main():
    #X_tr, X_te, y_tr, y_te = get_synth_data()
    synth_data = get_synth_data()
    clf = get_weight_prior(synth_data)#(X_tr) # bad name... getting classifier here
    y_train_pred = clf.labels_
    y_train_scores = clf.decision_scores_

    #y_test_pred = clf.predict(X_te)
    #y_test_scores = clf.decision_function(X_te)

    #print("\nOn Training Data:")
    #od.evaluate_print('LODA', y_tr, y_train_scores)
    #print("\nOn Test Data:")
    #od.evaluate_print('LODA', y_te, y_test_scores)

    #print(clf.histograms_)
    #print(synth_data.shape)
    #plt.imshow(clf.histograms_, cmap='hot', interpolation='nearest')
    #plt.show()
    #ax = sns.heatmap(clf.histograms_)
    avg = clf.histograms_.mean(axis=0)
    plt.plot(avg)
    plt.show()
    #visualize('LODA', X_tr, y_tr, X_te, y_te, y_train_pred,
          #y_test_pred, show_figure=True, save_figure=False)


if __name__ == '__main__':
    main()
