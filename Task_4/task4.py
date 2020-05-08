# %%
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

batch_size = 128

df = pd.read_csv(os.path.expanduser('train_triplets.txt'), sep=' ', names=['A', 'B', 'C'],
                 dtype='str')
df['label'] = 1
for index, row in df.iterrows():
    if index % 2 == 0:
        row['B'], row['C'] = row['C'], row['B']
        row['label'] = 0
        df.iloc[index] = row

train_df, test_df = train_test_split(df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

input_shape = (128, 128)


# %%
class train_dataset_cl(Sequence):
    def __init__(self, set, batch_size=64, shape=(64, 64)):
        """
        set is a dataframe
        """
        self.set = set
        self.batch_size = batch_size
        self.shape = shape
        self.done = False

    def __len__(self):
        return int(np.ceil(len(self.set) / float(self.batch_size)))

    def __getitem__(self, idx):
        x_batch = self.set.iloc[:, :-1].values[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_batch = self.set.iloc[:, -1].values[idx * self.batch_size:(idx + 1) * self.batch_size]
        x, y = [np.array(
            [resize(imread(os.path.join('food', '{}.jpg'.format(file_names[0]))), input_shape) for
             file_names in x_batch]), np.array(
            [resize(imread(os.path.join('food', '{}.jpg'.format(file_names[1]))), input_shape) for
             file_names in x_batch]), np.array(
            [resize(imread(os.path.join('food', '{}.jpg'.format(file_names[2]))), input_shape) for
             file_names in x_batch])], y_batch
        if not self.done:
            plt.subplot(1, 3, 1)
            plt.axis('off')
            plt.imshow(x[0][0])
            plt.subplot(1, 3, 2)
            plt.axis('off')
            plt.imshow(x[1][0])
            plt.subplot(1, 3, 3)
            plt.axis('off')
            plt.imshow(x[2][0])
            plt.savefig('temp/vis')
            self.done = True
        return x, y


train_dataset = train_dataset_cl(train_df, batch_size=batch_size, shape=input_shape)
val_dataset = train_dataset_cl(val_df, batch_size=batch_size, shape=input_shape)
test_dataset = train_dataset_cl(test_df, batch_size=batch_size, shape=input_shape)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, BatchNormalization, MaxPooling2D, concatenate, \
    Dropout, Activation, add, multiply, UpSampling2D, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

inputa = Input(shape=input_shape + (3,))
inputb = Input(shape=input_shape + (3,))
inputc = Input(shape=input_shape + (3,))


def attention_up_and_concate(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x


def head(x):
    x = Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape, padding="same")(x)
    x = Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape, padding="same")(x)

    gate = MaxPooling2D(pool_size=2, strides=2, data_format='channels_last')(x)
    gate = Conv2D(32, kernel_size=3, activation='relu', padding="same")(gate)
    x = attention_up_and_concate(gate, x)

    x = Conv2D(32, kernel_size=3, activation='relu', padding="same")(x)
    x = MaxPooling2D(pool_size=2)(x)

    return x


a = head(inputa)
b = head(inputb)
c = head(inputc)

x = concatenate([a, b, c])
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[inputa, inputb, inputc], outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# tf.keras.utils.plot_model(
#     model, to_file='model.png', show_shapes=False, show_layer_names=True,
#     rankdir='TB', expand_nested=False, dpi=96
# )

early_stopper = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint('models/', save_best_only=True)

model.fit(train_dataset, validation_data=val_dataset, epochs=100,
          steps_per_epoch=len(train_df) // batch_size,
          validation_steps=len(val_df) // batch_size, shuffle=True)
model.evaluate_generator(val_dataset, verbose=1)
