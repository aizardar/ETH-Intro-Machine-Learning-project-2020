import pandas as pd
import tensorflow as tf
import models
from models import simple_model, threelayers
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn import preprocessing, svm
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import os
import utils
import functools, operator
from models import dice_coef_loss
from tqdm import tqdm
from scipy.spatial.distance import dice

test = False
seed = 10
batch_size = 2048
num_subjects = -1         #number of subjects out of 18995
epochs = 1000

search_space_dict = {
    'loss': ['binary_crossentropy'],
    'nan_handling': ['minusone'],
    'standardizer': ['none'],
    'output_layer': ['sigmoid'],
    'model': ['threelayers'],
}

y_train_df = pd.read_csv('train_labels.csv').sort_values(by='pid')
y_train_df = y_train_df.iloc[:num_subjects, :10 + 1]
if not os.path.isfile('xtrain_imputedNN{}.csv'.format(num_subjects)):
    X_train_df = pd.read_csv('train_features.csv').sort_values(by='pid')
    X_train_df = X_train_df.loc[X_train_df['pid'] < y_train_df['pid'].values[-1] + 1]
    X_train_df = utils.impute_NN(X_train_df)
    # X_train_df['BaseExcess'].fillna(0)
    X_train_df.to_csv('xtrain_imputedNN{}.csv'.format(num_subjects), index = False)
else:
    X_train_df = pd.read_csv('xtrain_imputedNN{}.csv'.format(num_subjects))
