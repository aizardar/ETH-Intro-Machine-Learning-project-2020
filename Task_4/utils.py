import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence


# todo do I need to run Inception.preprocess?
class train_dataset_preprocessed(Sequence):
    def __init__(self, set, batches_save_path, dataset_type, batch_size=64, shape=(64, 64), save_dir='results'):
        """
        set is a dataframe
        """
        self.set = set
        self.batch_size = batch_size
        self.shape = shape
        self.done = False
        self.dataset_type = dataset_type
        self.batches_save_path = batches_save_path
        self.save_dir = save_dir

    def __len__(self):
        return int(np.ceil(len(self.set) / float(self.batch_size)))

    def __getitem__(self, idx):
        with open(
                os.path.join(self.batches_save_path, '{}batch_data'.format(self.dataset_type), 'xbatch_{}'.format(idx)),
                'rb') as fp:
            x_batch = pickle.load(fp)
        with open(
                os.path.join(self.batches_save_path, '{}batch_data'.format(self.dataset_type), 'ybatch_{}'.format(idx)),
                'rb') as fp:
            y_batch = pickle.load(fp)

        if not self.done:
            plt.figure(figsize=(10, 40))
            img_idx = 1
            for i in range(10):
                plt.subplot(10, 3, img_idx)
                plt.axis('off')
                plt.title(str(y_batch[i]))
                plt.imshow(x_batch[0][i])
                img_idx += 1
                plt.subplot(10, 3, img_idx)
                plt.axis('off')
                plt.imshow(x_batch[1][i])
                img_idx += 1
                plt.subplot(10, 3, img_idx)
                plt.axis('off')
                plt.imshow(x_batch[2][i])
                img_idx += 1
            plt.savefig(os.path.join(self.save_dir, 'vis'))
            self.done = True
        return x_batch, y_batch


class train_dataset_notpeprocessed(Sequence):
    def __init__(self, set, batch_size=64, shape=(64, 64), save_dir='results', with_keras_preprocessing=True):
        """
        set is a dataframe
        """
        self.set = set
        self.batch_size = batch_size
        self.shape = shape
        self.done = False
        self.save_dir = save_dir
        self.with_keras_preprocessing = with_keras_preprocessing

    def __len__(self):
        return int(np.ceil(len(self.set) / float(self.batch_size)))

    def __getitem__(self, idx):
        x_batch = self.set.iloc[:, :-1].values[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_batch = self.set.iloc[:, -1].values[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.with_keras_preprocessing:
            x, y = [np.array(
                [preprocess_input(resize(imread(os.path.join('food', '{}.jpg'.format(file_names[0]))), self.shape)*255) for
                 file_names in x_batch]), np.array(
                [preprocess_input(resize(imread(os.path.join('food', '{}.jpg'.format(file_names[1]))), self.shape)*255) for
                 file_names in x_batch]), np.array(
                [preprocess_input(resize(imread(os.path.join('food', '{}.jpg'.format(file_names[2]))), self.shape)*255) for
                 file_names in x_batch])], y_batch
        else:
            x, y = [np.array(
                [resize(imread(os.path.join('food', '{}.jpg'.format(file_names[0]))), self.shape) for
                 file_names in x_batch]), np.array(
                [resize(imread(os.path.join('food', '{}.jpg'.format(file_names[1]))), self.shape) for
                 file_names in x_batch]), np.array(
                [resize(imread(os.path.join('food', '{}.jpg'.format(file_names[2]))), self.shape) for
                 file_names in x_batch])], y_batch
        if not self.done:
            plt.figure(figsize=(10, 40))
            img_idx = 1
            for i in range(10):
                plt.subplot(10, 3, img_idx)
                plt.axis('off')
                plt.title(str(y_batch[i]))
                plt.imshow(x[0][i]*255)
                img_idx += 1
                plt.subplot(10, 3, img_idx)
                plt.axis('off')
                plt.imshow(x[1][i]*255)
                img_idx += 1
                plt.subplot(10, 3, img_idx)
                plt.axis('off')
                plt.imshow(x[2][i]*255)
                img_idx += 1
            plt.savefig(os.path.join(self.save_dir, 'vis'))
            self.done = True
        return x, y


class Pretr_inception():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.def_model()

    def def_model(self):
        inputa = Input(shape=self.input_shape + (3,))
        inputb = Input(shape=self.input_shape + (3,))
        inputc = Input(shape=self.input_shape + (3,))

        pretrained_modela = InceptionV3(input_tensor=inputa, include_top=False, weights='imagenet')
        for layer in pretrained_modela.layers:
            layer._name = layer.name + str("_a")
        for layer in pretrained_modela.layers:
            layer.trainable = False
        last_layera = pretrained_modela.get_layer('mixed7_a')
        last_outputa = last_layera.output

        pretrained_modelb = InceptionV3(input_tensor=inputb, include_top=False, weights='imagenet')
        for layer in pretrained_modelb.layers:
            layer._name = layer.name + str("_b")
        for layer in pretrained_modelb.layers:
            layer.trainable = False
        last_layerb = pretrained_modelb.get_layer('mixed7_b')
        last_outputb = last_layerb.output

        pretrained_modelc = InceptionV3(input_tensor=inputc, include_top=False, weights='imagenet')
        for layer in pretrained_modelc.layers:
            layer._name = layer.name + str("_c")
        for layer in pretrained_modelc.layers:
            layer.trainable = False
        last_layerc = pretrained_modelc.get_layer('mixed7_c')
        last_outputc = last_layerc.output

        x = concatenate([last_outputa, last_outputb, last_outputc])

        # Flatten the output layer to 1 dimension
        x = layers.Flatten()(x)
        # Add a fully connected layer with 1,024 hidden units and ReLU activation
        x = layers.Dense(1024, activation='relu')(x)
        # Add a dropout rate of 0.2
        x = layers.Dropout(0.2)(x)
        # Add a final sigmoid layer for classification
        x = layers.Dense(1, activation='sigmoid')(x)

        # Configure and compile the model
        self.model = Model([inputa, inputb, inputc], x)

    def fine_tuning(self):
        unfreeze = False

        for layer in self.model.layers:
            if unfreeze and not layer.name == 'flatten':
                layer.trainable = True
                print('set_to_trainable :', layer.name)

            if layer.name.startswith('mixed6'):
                unfreeze = True
            if layer.name == 'flatten':
                unfreeze = False
