import os
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from preprocess import store_batches
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import Sequence

batch_size = 128
file = 'train_triplets.txt'

df = pd.read_csv(file, sep=' ', names=['A', 'B', 'C'],
                 dtype='str')
df['label'] = 1

df_ = pd.read_csv(file, sep=' ', names=['B', 'A', 'C'],
                  dtype='str')
df_['label'] = 1

df = df.append(df_)

df_ = pd.read_csv(file, sep=' ', names=['A', 'C', 'B'],
                  dtype='str')
df_['label'] = 0

df = df.append(df_)

df_ = pd.read_csv(file, sep=' ', names=['B', 'C', 'A'],
                  dtype='str')
df_['label'] = 0

df = df.append(df_)
df = shuffle(df)
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)
train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=True)

input_shape = (128, 128)

if os.path.exists('/mnt/larry1/task_4_batches/'):
    batches_save_path = '/mnt/larry1/task_4_batches/'
elif os.path.exists('/media/miplab-nas2/Data/klug/hendrik'):
    batches_save_path = '/media/miplab-nas2/Data/klug/hendrik/task_4_batches/'
else:
    batches_save_path = os.path.expanduser('~/Desktop/larry1/task_4_batches/')

if not os.path.exists(batches_save_path):
    print('saving batches')
    os.mkdir(batches_save_path)
    store_batches(train_df, batches_save_path + 'train', input_shape, batch_size)
    store_batches(val_df, batches_save_path + 'val', input_shape, batch_size)
    store_batches(test_df, batches_save_path + 'test', input_shape, batch_size)


# %%
class train_dataset_cl(Sequence):
    def __init__(self, set, dataset_type, batch_size=64, shape=(64, 64)):
        """
        set is a dataframe
        """
        self.set = set
        self.batch_size = batch_size
        self.shape = shape
        self.done = False
        self.dataset_type = dataset_type

    def __len__(self):
        return int(np.ceil(len(self.set) / float(self.batch_size)))

    def __getitem__(self, idx):
        with open(os.path.join(batches_save_path, '{}batch_data'.format(self.dataset_type), 'xbatch_{}'.format(idx)),
                  'rb') as fp:
            x_batch = pickle.load(fp)
        with open(os.path.join(batches_save_path, '{}batch_data'.format(self.dataset_type), 'ybatch_{}'.format(idx)),
                  'rb') as fp:
            y_batch = pickle.load(fp)

        if not self.done:
            if not os.path.exists('temp'):
                os.mkdir('temp')
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
            plt.savefig('temp/vis')
            self.done = True
        return x_batch, y_batch


train_dataset = train_dataset_cl(train_df, 'train', batch_size=batch_size, shape=input_shape)
val_dataset = train_dataset_cl(val_df, 'val', batch_size=batch_size, shape=input_shape)
test_dataset = train_dataset_cl(test_df, 'test', batch_size=batch_size, shape=input_shape)

inputa = Input(shape=input_shape + (3,))
inputb = Input(shape=input_shape + (3,))
inputc = Input(shape=input_shape + (3,))

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
model = Model([inputa, inputb, inputc], x)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['acc'])

# tf.keras.utils.plot_model(
#     model, to_file='model.png', show_shapes=False, show_layer_names=True,
#     rankdir='TB', expand_nested=False, dpi=96
# )

early_stopper = EarlyStopping(monitor='val_acc', patience=5, verbose=True)
checkpoint_filepath = 'models/checkpoint.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, save_weights_only=True, verbose=True,
                             monitor='val_acc', mode='max')

# logger = tf.keras.callbacks.TensorBoard(log_dir='logs/',
#                                         histogram_freq=1,
#                                         profile_batch='500,520')

# model.fit(train_dataset, validation_data=val_dataset, epochs=20,
#           steps_per_epoch=len(train_df) // batch_size // 4,
#           validation_steps=len(val_df) // batch_size // 4, shuffle=True, callbacks=[early_stopper, checkpoint])
model.load_weights(checkpoint_filepath)

"""
"fine-tuning" the weights of the top layers of the pretrained models alongside the training of the top-level classifier
"""
from tensorflow.keras.optimizers import SGD

unfreeze = False

for model_name, model in zip(['_a', '_b', '_c'], [pretrained_modela, pretrained_modelb, pretrained_modelc]):
    # Unfreeze all models after "mixed6"
    for layer in model.layers:
        if unfreeze:
            layer.trainable = True
        if layer.name == 'mixed6' + model_name:
            print('unfreezing ', layer.name)
            unfreeze = True

# As an optimizer, here we will use SGD
# with a very low learning rate (0.00001)
model.compile(loss='binary_crossentropy',
              optimizer=SGD(
                  lr=0.00001,
                  momentum=0.9),
              metrics=['acc'])

checkpoint_filepath2 = 'models/checkpoint2.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, save_weights_only=True, verbose=True,
                             monitor='val_acc', mode='max')

model.fit(train_dataset, validation_data=val_dataset, epochs=100,
          steps_per_epoch=len(train_df) // batch_size // 4,
          validation_steps=len(val_df) // batch_size // 4, shuffle=True, callbacks=[early_stopper, checkpoint],
          verbose=2)

model.evaluate_generator(val_dataset, verbose=1)
