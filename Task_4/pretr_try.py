import os
import numpy as np
import pandas as pd
import tensorflow as tf
import utils
from preprocess import store_batches
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tqdm import tqdm

"""
Parameters
"""
save_dir = 'results'
with_keras_preprocessing = True
use_preprocessed = False
input_shape = (128, 128)
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
# train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)
train_df, val_df = train_test_split(df, test_size=0.3, shuffle=True)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
train_df.to_csv(os.path.join(save_dir, 'train_df.csv'), index=False)
val_df.to_csv(os.path.join(save_dir, 'val_df.csv'), index=False)

if use_preprocessed:
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
        # store_batches(test_df, batches_save_path + 'test', input_shape, batch_size)
    train_dataset = utils.train_dataset_preprocessed(train_df, 'train', batch_size=batch_size, shape=input_shape,
                                                     batches_save_path=batches_save_path)
    val_dataset = utils.train_dataset_preprocessed(val_df, 'val', batch_size=batch_size, shape=input_shape,
                                                   batches_save_path=batches_save_path)

else:
    train_dataset = utils.train_dataset_notpeprocessed(train_df, batch_size=batch_size, shape=input_shape,
                                                       save_dir=save_dir)
    val_dataset = utils.train_dataset_notpeprocessed(val_df, batch_size=batch_size, shape=input_shape,
                                                     save_dir=save_dir)
# %%

pretr_inception = utils.pretr_inception(input_shape)
model = pretr_inception.model

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['acc'])

# tf.keras.utils.plot_model(
#     model, to_file='model.png', show_shapes=False, show_layer_names=True,
#     rankdir='TB', expand_nested=False, dpi=96
# )

early_stopper = EarlyStopping(monitor='val_acc', patience=5, verbose=True)
checkpoint_filepath = os.path.join(save_dir, 'checkpoint.hdf5')
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, save_weights_only=True, verbose=True,
                             monitor='val_acc', mode='max')

logger = tf.keras.callbacks.TensorBoard(log_dir='logs/',
                                        histogram_freq=1,
                                        profile_batch='500,520')

model.fit(train_dataset, validation_data=val_dataset, epochs=20,
          steps_per_epoch=len(train_df) // batch_size // 4,
          validation_steps=len(val_df) // batch_size // 4, shuffle=True, callbacks=[early_stopper, checkpoint])
model.load_weights(checkpoint_filepath)
print('*****\n\n\n evaluation before fine-tuning \n\n\n*****')
model.evaluate_generator(val_dataset, verbose=1)
# %%
"""
"fine-tuning" the weights of the top layers of the pretrained results alongside the training of the top-level classifier
"""
from tensorflow.keras.optimizers import SGD

unfreeze = False

for layer in model.layers:
    if unfreeze and not layer.name == 'flatten':
        layer.trainable = True
        print('set_to_trainable :', layer.name)

    if layer.name.startswith('mixed6'):
        unfreeze = True
    if layer.name == 'flatten':
        unfreeze = False

# As an optimizer, here we will use SGD
# with a very low learning rate (0.00001)
model.compile(loss='binary_crossentropy',
              optimizer=SGD(
                  lr=0.00001,
                  momentum=0.9),
              metrics=['acc'])

checkpoint_filepath2 = os.path.join(save_dir, 'checkpoint2.hdf5')
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath2, save_best_only=True, save_weights_only=True, verbose=True,
                             monitor='val_acc', mode='max')

model.fit(train_dataset, validation_data=val_dataset, epochs=100,
          steps_per_epoch=len(train_df) // batch_size // 4,
          validation_steps=len(val_df) // batch_size // 4, shuffle=True, callbacks=[early_stopper, checkpoint],
          verbose=1)
model.load_weights(checkpoint_filepath2)
print('*****\n\n\n evaluation after fine-tuning \n\n\n*****')
model.evaluate_generator(val_dataset, verbose=1)

df_test = pd.read_csv('test_triplets.txt', sep=' ', names=['A', 'B', 'C'],
                      dtype='str')
for index, row in tqdm(df_test.iterrows()):
    images = [np.expand_dims(resize(imread(os.path.join('food', '{}.jpg'.format(row[col]))), input_shape), 0) for
              col in ['A', 'B', 'C']]
    df_test.at[index, 'label'] = int(np.round(model.predict(images), 0))

df_test.to_csv(os.path.join(save_dir, 'prediction.csv'), index=False)
df_pred = df_test[['label']].astype(int)
df_pred.to_csv(os.path.join(save_dir, 'prediction.txt'), header=None, index=None, sep=' ', mode='a')
