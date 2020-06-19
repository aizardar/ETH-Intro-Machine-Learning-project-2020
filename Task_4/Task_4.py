import pandas as pd
import numpy as np 
import tensorflow as tf
import os
from tqdm import tqdm
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte

# Set random seed

tf.random.set_seed(1234)



try:
    os.mkdir('resized_food_images')
except OSError:
    pass

"""
print("\nUnzipping images to directory resized_food_images")

local_zip = 'resized_food_images.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('resized_food_images')
zip_ref.close()

print("\nDone !")


"""


dir_path = os.path.join('food/')

print("\nResizing images to same size (256x256)..")

import imageio

for file in tqdm(os.listdir(dir_path)):
    if not file.startswith('.'):
        image = mpimg.imread(os.path.join(dir_path, '{}'.format(file)))
        resized_image = resize(image, (256, 256))
        imsave(os.path.join('resized_food_images', file),img_as_ubyte(resized_image))

print("\nDone !")

resized_dir_path = os.path.join('resized_food_images')
len(os.listdir(resized_dir_path))


files = os.listdir(resized_dir_path)
print(files[:10])


# Following code was to visualize the triplets 

"""
# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images\
pic_index = 0


# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 16
next_food_pic = [os.path.join(resized_dir_path, fname) 
                for fname in files[pic_index-16:pic_index]]



for i, img_path in enumerate(next_food_pic):
  # Set up subplot; subplot indices start at 1
    print(img_path)
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)
    img = mpimg.imread(img_path)
    print(img.shape)
    plt.imshow(img)

plt.show()

"""

df = pd.read_csv("train_triplets.txt", sep = " ", names = ["A","B","C"],dtype={'A': object,'B': object,'C': object})


new_A = []
new_B = []
new_C = []

for images in df["A"].values:
    new_A.append(os.path.join(resized_dir_path+"/"+str(images)+
                             ".jpg"))
for images in df["B"].values:
    new_B.append(os.path.join(resized_dir_path+"/"+str(images)+
                             ".jpg"))
for images in df["C"].values:
    new_C.append(os.path.join(resized_dir_path+"/"+str(images)+
                             ".jpg"))

split_size = 0.80
split_length = int(split_size*len(new_A))
train_A = new_A[:split_length]
train_B = new_B[:split_length]
train_C = new_C[:split_length]
val_A = new_A[split_length:]
val_B = new_B[split_length:]
val_C = new_C[split_length:]


train_data = {"A":train_A, "B":train_B,"C":train_C}
df_train = pd.DataFrame(train_data)
val_data = {"A":val_A, "B":val_B,"C":val_C}
df_val = pd.DataFrame(val_data)

# Function to feed batches of triplets

def data_generator(df, batch_size):
    
    while True:
        
        X_batch_A = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
        X_batch_B = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
        X_batch_C = np.empty((batch_size, 256, 256, 3), dtype=np.float32)

        for i in range(batch_size):

            im_A = os.path.join(df["A"].values[i])
            img_A = mpimg.imread(im_A)


            im_B = os.path.join(df["B"].values[i])
            img_B = mpimg.imread(im_B)


            im_C = os.path.join(df["C"].values[i])
            img_C = mpimg.imread(im_C)

            X_batch_A[i] = img_A / 255.
            X_batch_B[i] = img_B / 255.
            X_batch_C[i] = img_C / 255.

        yield [X_batch_A, X_batch_B, X_batch_C], np.empty((batch_size, 4096 * 3))



from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import InceptionV3

img_width = 256
img_height = 256
img_colors = 3

    
def triplet_loss(y_real, output):
    query, pos, neg = tf.unstack(tf.reshape(output, (-1, 3, 4096)), num=3, axis=1)

    positive_dist = tf.reduce_sum(tf.square(tf.subtract(query, pos)), 1)
    negative_dist = tf.reduce_sum(tf.square(tf.subtract(query, neg)), 1)
    
    loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), 0.6)
    loss = tf.reduce_mean(tf.maximum(loss_1, 0.0), 0)
    
    return loss    


@tf.function
def neg_euc_dist(y_true, output):

    query, pos, neg = tf.unstack(tf.reshape(output, (-1, 3, 4096)), num=3, axis=1)   
    
    negative_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(query, neg)), 1))
    return(tf.reduce_mean(negative_dist))


@tf.function
def pos_euc_dist(y_true, output):

    query, pos, neg = tf.unstack(tf.reshape(output, (-1, 3, 4096)), num=3, axis=1)   
        
    positive_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(query, pos)), 1))
    return(tf.reduce_mean(positive_dist))


def pretrained_convnet():
    
    base_model = InceptionV3(include_top=False, weights=None)
    
    local_weight_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    
    base_model.load_weights(local_weight_file)
    
    for layer in base_model.layers:
        layer.trainable = False

    x1 = base_model.output
    x1 = GlobalAveragePooling2D()(x1)
    x1 = Dense(4096, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x1)
    convnet_model = Model(inputs = base_model.input, outputs=x1)
    return convnet_model

    
def siamese_model(convnet_model):
    

    Input_1 = Input(shape=(img_width, img_height, img_colors))
    Input_2 = Input(shape=(img_width, img_height, img_colors))
    Input_3 = Input(shape=(img_width, img_height, img_colors))


    x1 = convnet_model(Input_1)
    x2 = convnet_model(Input_2)
    x3 = convnet_model(Input_3)
    
    concat_embed = Concatenate(axis=1, name='Concat_Embed')([x1,x2,x3])

    final_embed = Lambda(lambda  x: K.l2_normalize(x,axis=1))(concat_embed)

    model = Model([Input_1, Input_2, Input_3], final_embed)
    
    return model




convnet_model = pretrained_convnet()
train_model = siamese_model(convnet_model)

 
convnet_model.summary()

train_model.summary()

train_model.compile(optimizer= RMSprop(lr=0.000001), loss = triplet_loss, metrics = [pos_euc_dist,neg_euc_dist])

early_stopper = EarlyStopping(monitor='val_loss', patience = 2, verbose=True, mode = "min" , restore_best_weights = True)

checkpoint_file = 'checkpoint.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_file, save_best_only=True, save_weights_only=True, verbose=True, monitor='val_loss', mode='min')



history = train_model.fit(
    data_generator(df_train, batch_size=100),steps_per_epoch=476,
    validation_data=data_generator(df_val, batch_size=100),validation_steps=119,
    epochs = 50,callbacks=[early_stopper, checkpoint]) 

train_model.load_weights(checkpoint_file)

# Prediction on the validation set 

"""

prediction = []

for index, row in (df_val.iterrows()):
    
    #print(index)
    image_A = mpimg.imread(os.path.join(df_val["A"].values[index]))
    image_A = image_A / 255.
    image_A = image_A[tf.newaxis, ...]


    image_B = mpimg.imread(os.path.join(df_val["B"].values[index]))
    image_B = image_B / 255.
    image_B = image_B[tf.newaxis, ...]

  
    image_C =  mpimg.imread(os.path.join(df_val["C"].values[index]))
    image_C = image_C / 255.
    image_C = image_C[tf.newaxis, ...]
    
    embed_A = convnet_model(image_A)
    embed_B = convnet_model(image_B)
    embed_C = convnet_model(image_C)

    

    dist_A_B = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A, embed_B)), 1))
    dist_A_C = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A, embed_C)), 1))
    
    
    if dist_A_B < dist_A_C:
        
        prediction.append(1.)
    else:
        prediction.append(0.)


np.mean(prediction)

"""

print("\n Starting prediction on test dataset. This will take a while...\n")

df_test = pd.read_csv("test_triplets.txt", sep = " ", names = ["A","B","C"],dtype={'A': object,'B': object,'C': object})

new_A = []
new_B = []
new_C = []

for images in df_test["A"].values:
    new_A.append(os.path.join(resized_dir_path+"/"+str(images)+
                             ".jpg"))
for images in df_test["B"].values:
    new_B.append(os.path.join(resized_dir_path+"/"+str(images)+
                             ".jpg"))
for images in df_test["C"].values:
    new_C.append(os.path.join(resized_dir_path+"/"+str(images)+
                             ".jpg"))

test_data = {"A":new_A, "B":new_B,"C":new_C}

df_test = pd.DataFrame(test_data)


prediction_test = []


for index, row in tqdm(df_test.iterrows()):
    
    image_A = mpimg.imread(os.path.join(df_test["A"].values[index]))
    image_A = image_A / 255.
    image_A = image_A[tf.newaxis, ...]
    
    image_B = mpimg.imread(os.path.join(df_test["B"].values[index]))
    image_B = image_B / 255.
    image_B = image_B[tf.newaxis, ...]
    
    image_C =  mpimg.imread(os.path.join(df_test["C"].values[index]))
    image_C = image_C / 255.
    image_C = image_C[tf.newaxis, ...]
    
    
    embed_A_test = convnet_model(image_A)
    embed_B_test = convnet_model(image_B)
    embed_C_test = convnet_model(image_C)

    
    dist_A_B_test = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test, embed_B_test)), 1))
    dist_A_C_test = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test, embed_C_test)), 1))
    
    
    
    if dist_A_B_test < dist_A_C_test:
        
        prediction_test.append(1.)

    else:
        prediction_test.append(0.)

data = {"pred":prediction_test}
df = pd.DataFrame(data)
df = df.astype(int)
df.to_csv("predictions.dat",index = False, header = False)



"""



df_test_1 = df_test.iloc[:10000]
df_test_2 = df_test.iloc[10000:20000]
df_test_3 = df_test.iloc[20000:30000]
df_test_4 = df_test.iloc[30000:40000]
df_test_5 = df_test.iloc[40000:50000]
df_test_6 = df_test.iloc[50000:]


prediction_test_1 = []


for index, row in (df_test_1.iterrows()):
    
    image_A = mpimg.imread(os.path.join(df_test_1["A"].values[index]))
    image_A = image_A / 255.
    image_A = image_A[tf.newaxis, ...]
    
    image_B = mpimg.imread(os.path.join(df_test_1["B"].values[index]))
    image_B = image_B / 255.
    image_B = image_B[tf.newaxis, ...]
    
    image_C =  mpimg.imread(os.path.join(df_test_1["C"].values[index]))
    image_C = image_C / 255.
    image_C = image_C[tf.newaxis, ...]
    
    
    embed_A_test_1 = convnet_model(image_A)
    embed_B_test_1 = convnet_model(image_B)
    embed_C_test_1 = convnet_model(image_C)

    
    dist_A_B_test_1 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test_1, embed_B_test_1)), 1))
    dist_A_C_test_1 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test_1, embed_C_test_1)), 1))
    
    
    
    if dist_A_B_test_1 < dist_A_C_test_1:
        
        prediction_test_1.append(1.)

    else:
        prediction_test_1.append(0.)

data_1 = {"pred":prediction_test_1}
df_1 = pd.DataFrame(data_1)
df_1 = df_1.astype(int)
#df_1.to_csv("test_pred_1.dat",index = False, header = False)


df_test_2.reset_index(drop=True, inplace = True)


prediction_test_2 = []


for index, row in (df_test_2.iterrows()):
    
    image_A = mpimg.imread(os.path.join(df_test_2["A"].values[index]))
    image_A = image_A / 255.
    image_A = image_A[tf.newaxis, ...]
    image_B = mpimg.imread(os.path.join(df_test_2["B"].values[index]))
    image_B = image_B / 255.
    image_B = image_B[tf.newaxis, ...]
    
    image_C =  mpimg.imread(os.path.join(df_test_2["C"].values[index]))
    image_C = image_C / 255.
    image_C = image_C[tf.newaxis, ...]
    
    embed_A_test_2 = convnet_model(image_A)
    embed_B_test_2 = convnet_model(image_B)
    embed_C_test_2 = convnet_model(image_C)




    dist_A_B = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test_2, embed_B_test_2)), 1))
    dist_A_C = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test_2, embed_C_test_2)), 1))
    
    
    if dist_A_B < dist_A_C:
        
        prediction_test_2.append(1.)

    else:
        prediction_test_2.append(0.)

        
data_2 = {"pred":prediction_test_2}
df_2 = pd.DataFrame(data_2)
df_2 = df_2.astype(int)
#df_2.to_csv("test_pred_2.dat",index = False, header = False)


df_test_3.reset_index(drop=True, inplace = True)



prediction_test_3 = []


for index, row in (df_test_3.iterrows()):
    
    image_A = mpimg.imread(os.path.join(df_test_3["A"].values[index]))
    image_A = image_A / 255.
    image_A = image_A[tf.newaxis, ...]
    image_B = mpimg.imread(os.path.join(df_test_3["B"].values[index]))
    image_B = image_B / 255.
    image_B = image_B[tf.newaxis, ...]
    
    image_C =  mpimg.imread(os.path.join(df_test_3["C"].values[index]))
    image_C = image_C / 255.
    image_C = image_C[tf.newaxis, ...]
    
    embed_A_test_3 = convnet_model(image_A)
    embed_B_test_3 = convnet_model(image_B)
    embed_C_test_3 = convnet_model(image_C)




    dist_A_B = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test_3, embed_B_test_3)), 1))
    dist_A_C = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test_3, embed_C_test_3)), 1))
    
    
    if dist_A_B < dist_A_C:
        
        prediction_test_3.append(1.)

    else:
        prediction_test_3.append(0.)

data_3 = {"pred":prediction_test_3}
df_3 = pd.DataFrame(data_3)
df_3 = df_3.astype(int)
#df_3.to_csv("test_pred_3.dat",index = False, header = False)


df_test_4.reset_index(drop=True, inplace = True)


prediction_test_4 = []


for index, row in (df_test_4.iterrows()):
    

    image_A = mpimg.imread(os.path.join(df_test_4["A"].values[index]))
    image_A = image_A / 255.
    image_A = image_A[tf.newaxis, ...]
    image_B = mpimg.imread(os.path.join(df_test_4["B"].values[index]))
    image_B = image_B / 255.
    image_B = image_B[tf.newaxis, ...]
    
    image_C =  mpimg.imread(os.path.join(df_test_4["C"].values[index]))
    image_C = image_C / 255.
    image_C = image_C[tf.newaxis, ...]
    
    embed_A_test_4 = convnet_model(image_A)
    embed_B_test_4 = convnet_model(image_B)
    embed_C_test_4 = convnet_model(image_C)


    dist_A_B = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test_4, embed_B_test_4)), 1))
    dist_A_C = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test_4, embed_C_test_4)), 1))
    
    
    if dist_A_B < dist_A_C:
        
        prediction_test_4.append(1.)

    else:
        prediction_test_4.append(0.)

data_4 = {"pred":prediction_test_4}
df_4 = pd.DataFrame(data_4)
df_4 = df_4.astype(int)
#df_4.to_csv("test_pred_4.dat",index = False, header = False)


df_test_5.reset_index(drop=True, inplace = True)


prediction_test_5 = []


for index, row in (df_test_5.iterrows()):

    image_A = mpimg.imread(os.path.join(df_test_5["A"].values[index]))
    image_A = image_A / 255.
    image_A = image_A[tf.newaxis, ...]
    image_B = mpimg.imread(os.path.join(df_test_5["B"].values[index]))
    image_B = image_B / 255.
    image_B = image_B[tf.newaxis, ...]
    
    image_C =  mpimg.imread(os.path.join(df_test_5["C"].values[index]))
    image_C = image_C / 255.
    image_C = image_C[tf.newaxis, ...]
    
    embed_A_test_5 = convnet_model(image_A)
    embed_B_test_5 = convnet_model(image_B)
    embed_C_test_5 = convnet_model(image_C)


    dist_A_B = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test_5, embed_B_test_5)), 1))
    dist_A_C = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test_5, embed_C_test_5)), 1))
    
    
    if dist_A_B < dist_A_C:
        
        prediction_test_5.append(1.)

    else:
        prediction_test_5.append(0.)

data_5 = {"pred":prediction_test_5}
df_5 = pd.DataFrame(data_5)
df_5 = df_5.astype(int)
#df_5.to_csv("test_pred_5.dat",index = False, header = False)


df_test_6.reset_index(drop=True, inplace = True)


prediction_test_6 = []


for index, row in (df_test_6.iterrows()):
    

    image_A = mpimg.imread(os.path.join(df_test_6["A"].values[index]))
    image_A = image_A / 255.
    image_A = image_A[tf.newaxis, ...]
    image_B = mpimg.imread(os.path.join(df_test_6["B"].values[index]))
    image_B = image_B / 255.
    image_B = image_B[tf.newaxis, ...]
    
    image_C =  mpimg.imread(os.path.join(df_test_6["C"].values[index]))
    image_C = image_C / 255.
    image_C = image_C[tf.newaxis, ...]
    
    embed_A_test_6 = convnet_model(image_A)
    embed_B_test_6 = convnet_model(image_B)
    embed_C_test_6 = convnet_model(image_C)


    dist_A_B = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test_6, embed_B_test_6)), 1))
    dist_A_C = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(embed_A_test_6, embed_C_test_6)), 1))
    
    
    if dist_A_B < dist_A_C:
        
        prediction_test_6.append(1.)

    else:
        prediction_test_6.append(0.)
        
data_6 = {"pred":prediction_test_6}
df_6 = pd.DataFrame(data_6)
df_6 = df_6.astype(int)
#df_6.to_csv("test_pred_6.dat",index = False, header = False)


df_1 = pd.read_csv("test_pred_1.dat", header = None)
df_1.reset_index(drop=True, inplace = True)
df_2 = pd.read_csv("test_pred_2.dat", header = None)
df_2.reset_index(drop=True, inplace = True)
df_3 = pd.read_csv("test_pred_3.dat", header = None)
df_2.reset_index(drop=True, inplace = True)
df_4 = pd.read_csv("test_pred_4.dat", header = None)
df_4.reset_index(drop=True, inplace = True)
df_5 = pd.read_csv("test_pred_5.dat", header = None)
df_5.reset_index(drop=True, inplace = True)
df_6 = pd.read_csv("test_pred_6.dat", header = None)
df_6.reset_index(drop=True, inplace = True)


df_combined = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6], ignore_index=True, sort = False)

df_combined.to_csv("predictions.dat", index= False, header=False)

"""

print("\n All Done !\n")
