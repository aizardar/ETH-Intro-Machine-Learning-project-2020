import pandas as pd
import tensorflow as tf
from models import simple_model
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from matplotlib import pyplot as plt


"""
Predict whether medical tests are ordered by a clinician in the remainder of the hospital stay: 0 means that there will be no further tests of this kind ordered, 1 means that at least one of a test of that kind will be ordered. In the submission file, you are asked to submit predictions in the interval [0, 1], i.e., the predictions are not restricted to binary. 0.0 indicates you are certain this test will not be ordered, 1.0 indicates you are sure it will be ordered. The corresponding columns containing the binary groundtruth in train_labels.csv are: LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2.
10 labels for this subtask
"""

seed = 100
batch_size = 1
num_subjects = 100          #number of subjects out of 18995
loss = 'categorical_crossentropy'


X_train_df = pd.read_csv('train_features.csv').sort_values(by= 'pid')
y_train_df = pd.read_csv('train_labels.csv').sort_values(by= 'pid')
y_train = y_train_df.iloc[:num_subjects, :10 + 1]

X_train_df = X_train_df.loc[X_train_df['pid'] < y_train['pid'].values[-1] + 1]

"""
instead of imputing I could also set all the nans to -1
"""
imp = IterativeImputer(max_iter=10, random_state=seed)
imp.fit(X_train_df)
x_train_imp = pd.DataFrame(data = imp.transform(X_train_df), columns = X_train_df.columns)

scaler = preprocessing.StandardScaler()
x_train_df = pd.DataFrame(data = scaler.fit_transform(x_train_imp.values[:, 1:]), columns= X_train_df.columns[1:])
x_train_df.insert(0, 'pid', X_train_df['pid'].values)
x_train_df.to_csv('temp/taining_data.csv')

x_train = []
for i, subject in enumerate(list(dict.fromkeys(x_train_df['pid'].values.tolist()))):
    x_train.append(np.concatenate(x_train_imp.loc[x_train_imp['pid'] == subject].values[:, 1:]))
input_shape = x_train[0].shape
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train.values[:, 1:], test_size=0.25, random_state=seed)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(num_subjects).batch(batch_size=batch_size).repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.shuffle(num_subjects).batch(batch_size=batch_size).repeat()
model = simple_model(input_shape, loss)
history = model.fit(train_dataset, validation_data = val_dataset, epochs = 10, steps_per_epoch=input_shape[0]//batch_size, validation_steps = input_shape[0]//batch_size)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


print(model.summary())
print('\nhistory dict:', history.history)

prediction = model.predict(tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size=batch_size))
loss, accuracy = model.evaluate(train_dataset, steps=input_shape[0]//batch_size)
dsf = 0