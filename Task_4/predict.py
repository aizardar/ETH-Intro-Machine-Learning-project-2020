import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tqdm import tqdm

df_test = pd.read_csv('test_triplets.txt', sep=' ', names=['A', 'B', 'C'],
                      dtype='str')

print(len(df_test))

df_test['label'] = np.nan
input_shape = (128, 128)

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

model.load_weights('models/checkpoint.hdf5')
for index, row in tqdm(df_test.iterrows()):
    images = [np.expand_dims(resize(imread(os.path.join('food', '{}.jpg'.format(row[col]))), input_shape), 0) for
              col in ['A', 'B', 'C']]
    df_test.at[index, 'label'] = int(np.round(model.predict(images), 0))

df_test.to_csv('prediction.csv', index=False)
df_pred = df_test[['label']].astype(int)
df_pred.to_csv('prediction.txt', header=None, index=None, sep=' ', mode='a')
