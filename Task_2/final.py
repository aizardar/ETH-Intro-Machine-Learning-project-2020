import pandas as pd
import utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
import models
import numpy as np
import sklearn.metrics as metrics
from utils import scaling
from sklearn.linear_model import LinearRegression
from models import dice_coef_loss

SEED = 1
batch_size = 2048

"""
Loading Data
"""
final_df = pd.read_csv('sample.csv')
y_train_df = pd.read_csv('train_labels.csv').sort_values(by='pid')
X_train_df = pd.read_csv('train_features.csv').sort_values(by='pid')
X_final_df = pd.read_csv('test_features.csv')
"""
Subtask 1 and 2
"""
# X_train_df = pd.read_csv('imputed_data/xtrain_imputedNN{}.csv'.format(-1))
# X_final_df = pd.read_csv('imputed_data/xtest_imputedNN{}.csv'.format(-1))
X_final_df = utils.impute_NN(X_final_df)
X_train_df = utils.impute_NN(X_train_df)
params = {'nan_handling': 'iterative', 'collapse_time':'yes'}
X_train_df = utils.handle_nans(X_train_df, params, SEED)
X_final_df = utils.handle_nans(X_final_df, params, SEED)
x_train = X_train_df.groupby('pid').mean().reset_index().values[:, 1:]
x_final = X_final_df.groupby('pid').mean().reset_index().values[:, 1:]

y_train1 = list(y_train_df.values[:, 1:12])
x_train, x_valtest, y_train, y_valtest = train_test_split(x_train, y_train1, test_size=0.4, random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_valtest, y_valtest, test_size=0.3, random_state=SEED)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size=batch_size).repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.shuffle(len(x_val)).batch(batch_size=batch_size).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
test_dataset = test_dataset.batch(batch_size=1)
"""
Callbacks
"""
CB_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                             patience=5,
                                             verbose=1,
                                             min_delta=0.0001,
                                             min_lr=1e-6)

CB_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=0.0001,
                                         verbose=1,
                                         patience=10,
                                         mode='min',
                                         restore_best_weights=True)
callbacks = [CB_es, CB_lr]

model = models.threelayers_task1(x_train.shape, dice_coef_loss, 'sigmoid', 1)
model.fit(train_dataset, validation_data=val_dataset, epochs=1000, steps_per_epoch=len(x_train)//2048,
                      validation_steps=len(x_train)//2048, callbacks=callbacks)


prediction = model.predict(test_dataset)
test_prediction_df = pd.DataFrame(prediction, columns=y_train_df.columns[1:12])
y_test_df = pd.DataFrame(np.vstack(y_test), columns=y_train_df.columns[1:12])
roc_auc = [metrics.roc_auc_score(y_test_df[entry], test_prediction_df[entry]) for entry in y_train_df.columns[1:12]]
score = np.mean(roc_auc)
print(roc_auc, 'score:', score)


final_dataset = tf.data.Dataset.from_tensor_slices(x_final)
final_dataset = final_dataset.batch(batch_size=1)

prediction1 = model.predict(final_dataset)
prediction_df = pd.DataFrame(prediction1, columns=y_train_df.columns[1:12])
for col in prediction_df.columns:
    final_df[col] = prediction_df[col]
final_df.reindex(columns=final_df.columns)

"""
Subtask 3
"""
y_train_df = pd.read_csv('train_labels.csv').sort_values(by='pid')
X_train_df = pd.read_csv('train_features.csv').sort_values(by='pid')
X_final_df = pd.read_csv('test_features.csv').sort_values(by = 'pid')

params = {'nan_handling': 'iterative', 'collapse_time':'yes', 'standardizer':'minmax'}
X_train_df = utils.handle_nans(X_train_df, params, SEED)
X_final_df = utils.handle_nans(X_final_df, params, SEED)
x_train = X_train_df.groupby('pid').mean().reset_index().values[:, 2:]
x_final = X_final_df.groupby('pid').mean().reset_index().values[:, 2:]
y_train3 = list(y_train_df.values[:, 12:])

x_train, x_valtest, y_train, y_valtest = train_test_split(x_train, y_train3, test_size=0.4, random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_valtest, y_valtest, test_size=0.3, random_state=SEED)

x_train, x_scaler = scaling(x_train, params)
x_val, _ = scaling(x_val, params, x_scaler)
x_test, _ = scaling(x_test, params, x_scaler)

model = LinearRegression().fit(x_train, y_train)
prediction = model.predict(x_test)
prediction_df = pd.DataFrame(prediction, columns=y_train_df.columns[12:])
y_test_df = pd.DataFrame(np.vstack(y_test), columns=y_train_df.columns[12:])
print('r2_scores: ', [metrics.r2_score(y_test_df[entry], prediction_df[entry]) for entry in y_train_df.columns[12:]])

x_final, _ = utils.scaling(x_final, params, x_scaler)
prediction3 = model.predict(x_final)
prediction_df = pd.DataFrame(prediction3, columns=y_train_df.columns[12:])
for col in prediction_df.columns:
    final_df[col] = prediction_df[col]
final_df.reindex(columns=final_df.columns)
sample = pd.read_csv('sample.csv')
final_df.to_csv('prediction.csv', index = False, float_format='%.3f')
final_df.to_csv('prediction.zip', index = False, float_format='%.3f', compression='zip')
