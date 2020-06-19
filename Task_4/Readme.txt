For this task, I have assumed that all the images are already extracted in the folder 'food'. Code will read images from the folder and will create another folder named 'resized_food_images' to store the resized images for faster data loading during training.

Code requires following files in the current directory:

1. train_triplet.txt
2. test_triplet.txt
3. weights file - inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 (This file can be downloaded from "https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")
