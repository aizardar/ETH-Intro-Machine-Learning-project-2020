from tensorflow.keras.utils import Sequence
import pandas as pd
import os
from tqdm import tqdm

from skimage.io import imread, imsave
from skimage.transform import resize

for file in tqdm(os.listdir('food')):
    if not file.startswith('.'):
        x = resize(imread(os.path.join('food', '{}'.format(file))), (128, 128))
        imsave(os.path.join('food_resized', file), x)
