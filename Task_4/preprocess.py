import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
import pickle
import os
from tqdm import tqdm

def store_batches(set, data_set, input_shape, batch_size):
    for idx in tqdm(range(int(np.ceil(len(set) / float(batch_size))))):
        x_batch = set.iloc[:, :-1].values[idx * batch_size:(idx + 1) *batch_size]
        y_batch = set.iloc[:, -1].values[idx * batch_size:(idx + 1) * batch_size]
        x, y = [np.array(
            [resize(imread(os.path.join('food', '{}.jpg'.format(file_names[0]))), input_shape) for
             file_names in x_batch]), np.array(
            [resize(imread(os.path.join('food', '{}.jpg'.format(file_names[1]))), input_shape) for
             file_names in x_batch]), np.array(
            [resize(imread(os.path.join('food', '{}.jpg'.format(file_names[2]))), input_shape) for
             file_names in x_batch])], y_batch
        if not os.path.exists('{}batch_data'.format(data_set)):
            os.mkdir('{}batch_data'.format(data_set))
        with open('{}batch_data/xbatch_{}'.format(data_set, idx), 'wb') as fp:
            pickle.dump(x, fp)
        with open('{}batch_data/ybatch_{}'.format(data_set, idx), 'wb') as fp:
            pickle.dump(y, fp)
