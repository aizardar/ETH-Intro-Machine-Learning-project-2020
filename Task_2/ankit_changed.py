#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import sklearn
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# In[2]:
SEED = 42

train_features = pd.read_csv("train_features.csv").sort_values(by='pid')
test = pd.read_csv("test_features.csv")
train_labels = pd.read_csv("train_labels.csv").sort_values(by='pid')


# In[3]:


# Let's get rid of Hgb, HCO3, ABPd, and Bilirubin_direct from the training and test data !

# train_features.drop('Bilirubin_direct', axis=1, inplace=True)
# train_features.drop('HCO3', axis=1, inplace=True)
# train_features.drop('Hgb', axis=1, inplace=True)
# train_features.drop('ABPd', axis=1, inplace=True)
#
# test.drop('Bilirubin_direct', axis=1, inplace=True)
# test.drop('HCO3', axis=1, inplace=True)
# test.drop('Hgb', axis=1, inplace=True)
# test.drop('ABPd', axis=1, inplace=True)


# In[4]:


def impute_KNN(df):
    imputed_df = pd.DataFrame(columns=df.columns)
    imp = sklearn.impute.KNNImputer(n_neighbors=2)
    for pid in tqdm(np.unique(df['pid'].values)):
        temp_df = df.loc[df['pid'] == pid]
        temp_df2 = temp_df.dropna(axis='columns', how='all')
        imp.fit(temp_df2)
        temp_df2 = pd.DataFrame(data=imp.transform(temp_df2), columns=temp_df2.columns)
        for key in temp_df.columns:
            if temp_df[key].isna().all():
                temp_df2[key] = np.nan
        imputed_df = imputed_df.append(temp_df2, sort=True)
    imputed_df.reindex(columns=df.columns)
    return imputed_df


# In[5]:

train_path = 'nan_handling/train.csv'
test_path = 'nan_handling/test.csv'
if os.path.isfile(train_path) and os.path.isfile(test_path):
    train_features_imp = pd.read_csv(train_path)
    test_imputed = pd.read_csv(test_path)
else:
    train_features_imp = impute_KNN(train_features)
    train_features_imp.to_csv(train_path, index=False)
    test_imputed = impute_KNN(test)
    test_imputed.to_csv(test_path, index=False)


# train_features_imp = train_features
# test_imputed = test
# In[6]:


def impute_mode(df):
    for column in df.columns:
        # df[column].fillna(df[column].mode()[0], inplace=True)
        df[column].fillna(-1, inplace=True)

    return df


# In[7]:


train_features_ = impute_mode(train_features_imp)
test_ = impute_mode(test_imputed)

# In[8]:


# Let's now take the mean of all pids so that we have just one row per patient

train_features = train_features_.groupby('pid').mean().reset_index()
test = test_.groupby('pid').mean().reset_index()
# %% split into personal test set and train set
train_features, x_personal_test, train_labels, y_personal_test = train_test_split(train_features, train_labels,
                                                                                  test_size=0.2, random_state=SEED)

# In[9]:

# Normalize the input features using the sklearn StandardScaler. This will set the mean to 0 and standard deviation to 1.

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=train_features.columns)

X_p_test_scaled = scaler.transform(x_personal_test)
X_p_test_scaled = pd.DataFrame(X_p_test_scaled, columns=x_personal_test.columns)

X_test_scaled = scaler.transform(test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=test.columns)

# In[10]:

feature_cols = X_train_scaled.columns.values[
    (X_train_scaled.columns.values != 'pid') & (X_train_scaled.columns.values != 'Time')]

X_train_scaled_ = X_train_scaled[feature_cols]
X_test_scaled_ = X_test_scaled[feature_cols]
X_p_test_scaled_ = X_p_test_scaled[feature_cols]

# In[11]:


# # Task 1 and Task 2
#
# from sklearn.svm import SVC
#
# TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
#          'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
#          'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis']
#
# prediction_df = pd.DataFrame(test.pid.values, columns=['pid'])
# p_test_prediction_df = pd.DataFrame(x_personal_test['pid'].values, columns=['pid'])
#
# for label in tqdm(TESTS):
#
#     if label == 'LABEL_BaseExcess':
#
#         y_train_temp = train_labels[label]
#         model = SVC(kernel='rbf', gamma=0.01, C=10, random_state=42, probability=True)
#         model.fit(X_train_scaled, y_train_temp)
#
#         p_test_pred = model.predict_proba(X_p_test_scaled)
#         p_test_pred_ = pd.DataFrame(np.ravel(p_test_pred[:,1]), columns=[label])
#         p_test_prediction_df = pd.concat([p_test_prediction_df, p_test_pred_], axis =1, sort = False)
#
#         predictions_prob_test = model.predict_proba(X_test_scaled)
#         predict_prob_test = pd.DataFrame(np.ravel(predictions_prob_test[:, 1]), columns=[label])
#         prediction_df = pd.concat([prediction_df, predict_prob_test], axis=1, sort=False)
#
#     elif label == 'LABEL_TroponinI':
#
#         y_train_temp = train_labels[label]
#         model = SVC(kernel='rbf', gamma=0.001, C=1, random_state=42, probability=True)
#         model.fit(X_train_scaled, y_train_temp)
#
#         p_test_pred = model.predict_proba(X_p_test_scaled)
#         p_test_pred_ = pd.DataFrame(np.ravel(p_test_pred[:,1]), columns=[label])
#         p_test_prediction_df = pd.concat([p_test_prediction_df, p_test_pred_], axis =1, sort = False)
#
#         predictions_prob_test = model.predict_proba(X_test_scaled)
#         predict_prob_test = pd.DataFrame(np.ravel(predictions_prob_test[:, 1]), columns=[label])
#         prediction_df = pd.concat([prediction_df, predict_prob_test], axis=1, sort=False)
#
#     elif label == 'LABEL_AST':
#
#         y_train_temp = train_labels[label]
#         model = SVC(kernel='rbf', gamma=0.001, C=1, random_state=42, probability=True)
#         model.fit(X_train_scaled, y_train_temp)
#
#         p_test_pred = model.predict_proba(X_p_test_scaled)
#         p_test_pred_ = pd.DataFrame(np.ravel(p_test_pred[:,1]), columns=[label])
#         p_test_prediction_df = pd.concat([p_test_prediction_df, p_test_pred_], axis =1, sort = False)
#
#         predictions_prob_test = model.predict_proba(X_test_scaled)
#         predict_prob_test = pd.DataFrame(np.ravel(predictions_prob_test[:, 1]), columns=[label])
#         prediction_df = pd.concat([prediction_df, predict_prob_test], axis=1, sort=False)
#
#     else:
#
#         y_train_temp = train_labels[label]
#         model = SVC(kernel='rbf', gamma=0.1, C=1, random_state=42, probability=True)
#         model.fit(X_train_scaled, y_train_temp)
#
#         p_test_pred = model.predict_proba(X_p_test_scaled)
#         p_test_pred_ = pd.DataFrame(np.ravel(p_test_pred[:,1]), columns=[label])
#         p_test_prediction_df = pd.concat([p_test_prediction_df, p_test_pred_], axis =1, sort = False)
#
#         predictions_prob_test = model.predict_proba(X_test_scaled)
#         predict_prob_test = pd.DataFrame(np.ravel(predictions_prob_test[:, 1]), columns=[label])
#         prediction_df = pd.concat([prediction_df, predict_prob_test], axis=1, sort=False)


# %% Define model:
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def threelayers(input_shape, loss, output_layer, task=1):
    model = keras.Sequential()

    model.add(keras.layers.BatchNormalization(axis=-1))
    # Define first fully connected layer
    model.add(keras.layers.Dense(400,
                                 input_shape=input_shape,
                                 activation=tf.nn.relu,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(l=1e-3)))

    # Add dropout for overfitting avoidance and batch normalization layer
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.BatchNormalization(scale=False))

    # Add second fully connected layer
    model.add(keras.layers.Dense(250,
                                 activation=tf.nn.relu,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(l=1e-3)))

    # Add dropout for overfitting avoidance and batch normalization layer
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.BatchNormalization(scale=False))
    model.add(keras.layers.Flatten())
    # Add output layer
    if task == 1:
        model.add(keras.layers.Dense(10, activation=output_layer))
    if task == 2:
        model.add(keras.layers.Dense(1, activation=output_layer))
    model.compile(optimizer='adagrad',
                  loss=loss,
                  metrics=[dice_coef, 'mse', keras.metrics.AUC()])

    return model


# %% Datasets and callbacks:

batch_size = 2048

train_labels_task1 = train_labels.iloc[:, 1:11]
p_test_labels_task1 = y_personal_test.iloc[:, 1:11]

x_train, x_val, y_train, y_val = train_test_split(X_train_scaled_, train_labels_task1, test_size=0.2, random_state=SEED)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))
train_dataset = train_dataset.shuffle(len(X_train_scaled_)).batch(batch_size=batch_size).repeat()

p_test_dataset = tf.data.Dataset.from_tensor_slices((X_p_test_scaled_.values, p_test_labels_task1.values))
p_test_dataset = p_test_dataset.batch(batch_size=batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val.values, y_val.values))
val_dataset = val_dataset.shuffle(len(x_val)).batch(batch_size=batch_size).repeat()

test_dataset = tf.data.Dataset.from_tensor_slices(X_test_scaled_.values)
test_dataset = test_dataset.batch(batch_size=batch_size)

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

# %% Train model
print('\n\n**** Task 1 **** \n\n')

model = threelayers(x_train.shape, dice_coef_loss, 'sigmoid')
model.fit(train_dataset, validation_data=val_dataset, epochs=1, steps_per_epoch=len(x_train) // 2048,
          validation_steps=len(x_train) // 2048, callbacks=callbacks)

# %% Predict on personal test-set:
model.evaluate(p_test_dataset)
p_test_prediction = model.predict(p_test_dataset)
p_test_prediction_df = pd.DataFrame(p_test_prediction, columns=train_labels.columns[1:11])
p_test_prediction_df['pid'] = y_personal_test['pid'].values

# %% Predict on test-set:
prediction = model.predict(test_dataset)
prediction_df = pd.DataFrame(prediction, columns=train_labels.columns[1:11])
prediction_df['pid'] = test.pid.values

# In[ ]:
#
#
# %% Task 2

print('\n\n**** Task 2 **** \n\n')
train_labels_task2 = train_labels.iloc[:, 11]
p_test_labels_task2 = y_personal_test.iloc[:, 11]

x_train, x_val, y_train, y_val = train_test_split(X_train_scaled_, train_labels_task2, test_size=0.2, random_state=SEED)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))
train_dataset = train_dataset.shuffle(len(X_train_scaled_)).batch(batch_size=batch_size).repeat()

p_test_dataset = tf.data.Dataset.from_tensor_slices((X_p_test_scaled_.values, p_test_labels_task2.values))
p_test_dataset = p_test_dataset.batch(batch_size=batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val.values, y_val.values))
val_dataset = val_dataset.shuffle(len(x_val)).batch(batch_size=batch_size).repeat()

model = threelayers(x_train.shape, dice_coef_loss, 'sigmoid', task=2)
model.fit(train_dataset, validation_data=val_dataset, epochs=1, steps_per_epoch=len(x_train) // 2048,
          validation_steps=len(x_train) // 2048, callbacks=callbacks)

model.evaluate(p_test_dataset)
p_test_prediction_task2 = model.predict(p_test_dataset)
p_test_prediction_df[train_labels.columns[11]] = p_test_prediction_task2

prediction = model.predict(test_dataset)
prediction_df[train_labels.columns[11]] = prediction

# %%
from ignite.contrib.metrics.regression import R2Score
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import tensorboard


class testDataset(Dataset):
    def __init__(self, data):
        super(testDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.from_numpy(self.data[item][None, ...]).float()


class MyDataset(Dataset):
    def __init__(self, data, labels):
        super(MyDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.from_numpy(self.data[item][None, ...]).float(), torch.from_numpy(self.labels[item]).float()


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Net(LightningModule):
    def __init__(self, train_dataset, val_dataset, trial):
        super(Net, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        kernel_size1 = trial.suggest_int('kernel_size1', 2, 12)
        kernel_size2 = trial.suggest_int('kernel_size2', 2, 35)
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size= (kernel_size1, kernel_size2), stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d((2, 1)),
            Flatten(),
            torch.nn.Linear(11200, 50),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(50, 4)
        )
        self.r2_score1 = R2Score()
        self.r2_score2 = R2Score()
        self.r2_score3 = R2Score()
        self.r2_score4 = R2Score()

    def forward(self, x):
        return self.network(x)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=2048, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=2048, num_workers=16)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_index):
        data, target = batch
        output = self.forward(data)
        loss = self.mse(output, target)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_index):
        data, target = batch
        output = self.forward(data)
        loss = self.mse(output, target)
        self.r2_score1.update((output[:, 0], target[:, 0]))
        self.r2_score2.update((output[:, 1], target[:, 1]))
        self.r2_score3.update((output[:, 2], target[:, 2]))
        self.r2_score4.update((output[:, 3], target[:, 3]))
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_r2_score1 = self.r2_score1.compute()
        self.r2_score1.reset()
        avg_r2_score2 = self.r2_score2.compute()
        self.r2_score2.reset()
        avg_r2_score3 = self.r2_score3.compute()
        self.r2_score3.reset()
        avg_r2_score4 = self.r2_score4.compute()
        self.r2_score4.reset()
        avg_r2 = (avg_r2_score1 + avg_r2_score2 + avg_r2_score3 + avg_r2_score4) / 4

        tensorboard_logs = {'val_loss': avg_loss, 'val_r2_score1': avg_r2_score1, 'val_r2_score2': avg_r2_score2,
                            'val_r2_score3': avg_r2_score3, 'val_r2_score4': avg_r2_score4, 'avg_r2': avg_r2, }
        print(avg_r2)
        return {'val_loss': avg_loss, 'val_r2_score': avg_r2, 'log': tensorboard_logs}

    def mse(self, logits, labels):
        return torch.nn.functional.mse_loss(logits, labels)

#%%
# import pytorch_lightning as pl
# from pytorch_lightning.logging import LightningLoggerBase
# import optuna
# from optuna.integration import PyTorchLightningPruningCallback
#
# 
# if not os.path.exists('optunaaa'):
#     os.mkdir('optunaaa')
#
# EPOCHS = 100
# DIR = 'optunaaa'
# MODEL_DIR = os.path.join(DIR, "result")
#
# def objective(trial):
#     # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
#     # filenames match. Therefore, the filenames for each trial must be made unique.
#     checkpoint_callback = pl.callbacks.ModelCheckpoint(
#         os.path.join(MODEL_DIR, "trial_{}".format(trial.number)), monitor="val_r2_score"
#     )
#
#     # The default logger in PyTorch Lightning writes to event files to be consumed by
#     # TensorBoard. We create a simple logger instead that holds the log in memory so that the
#     # final accuracy can be obtained after optimization. When using the default logger, the
#     # final accuracy could be stored in an attribute of the `Trainer` instead.
#     logger = DictLogger(trial.number)
#
#     trainer = pl.Trainer(
#         logger=logger,
#         checkpoint_callback=checkpoint_callback,
#         max_epochs=EPOCHS,
#         gpus=0 if torch.cuda.is_available() else None,
#         early_stop_callback= PyTorchLightningPruningCallback(trial, monitor="val_r2_score"),
#     )
#     model = Net(trial)
#     trainer.fit(model)
#     return logger.metrics[-1]["val_r2_score"]
#
# class DictLogger(LightningLoggerBase):
#     """PyTorch Lightning `dict` logger."""
#
#     def __init__(self, version):
#         super(DictLogger, self).__init__()
#         self.metrics = []
#         self._version = version
#
#     def log_metrics(self, metric, step=None):
#         self.metrics.append(metric)
#
#     @property
#     def version(self):
#         return self._version
#
# pruner = optuna.pruners.MedianPruner()
#
# study = optuna.create_study(direction="maximize", pruner=pruner)
# study.optimize(objective, n_trials=100, timeout=600)
#
# print("Number of finished trials: {}".format(len(study.trials)))
#
# print("Best trial:")
# trial = study.best_trial
#
# print("  Value: {}".format(trial.value))
#
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))

# %% Task3 with multiple output CNN
"""
Need to take data from before scaling and scale it afterwards
"""
# train_labels_ = pd.read_csv("train_labels.csv").sort_values(by='pid')
train_labels_task3 = list(train_labels.values[:, 12:])

x_train = []
for i, subject in enumerate(list(dict.fromkeys(train_features['pid'].values.tolist()))):
    x_train.append(train_features_.loc[train_features_['pid'] == subject, feature_cols].values)

x_personal_test_3 = []
for i, subject in enumerate(list(dict.fromkeys(x_personal_test['pid'].values.tolist()))):
    x_personal_test_3.append(train_features_.loc[train_features_['pid'] == subject, feature_cols].values)

x_test_scaled_task3 = []
for i, subject in enumerate(list(dict.fromkeys(test_['pid'].values.tolist()))):
    x_test_scaled_task3.append(test_.loc[test_['pid'] == subject, feature_cols].values)

scaler = RobustScaler()

x_temp = np.concatenate(x_train)
x_temp = scaler.fit_transform(x_temp)
x_train = np.vsplit(x_temp, x_temp.shape[0] / 12)

x_temp = np.concatenate(x_personal_test_3)
x_temp = scaler.transform(x_temp)
x_personal_test_3 = np.vsplit(x_temp, x_temp.shape[0] / 12)

x_temp = np.concatenate(x_test_scaled_task3)
x_temp = scaler.transform(x_temp)
x_test_scaled_task3 = np.vsplit(x_temp, x_temp.shape[0] / 12)

input_shape = x_train[0].shape

x_train, x_val, y_train, y_val = train_test_split(x_train, train_labels_task3, test_size=0.2, random_state=SEED)

train_dataset = MyDataset(x_train, y_train)
val_dataset = MyDataset(x_val, y_val)
personal_test_dataset = testDataset(x_personal_test)
logger = TensorBoardLogger("tb_logs", name="task3")
checkpoint_callback = ModelCheckpoint(filepath='temp/', verbose=True)
model = Net(train_dataset, val_dataset)
trainer = Trainer(checkpoint_callback=checkpoint_callback, logger=logger, profiler=True)
trainer.fit(model)

p_test_prediction_task3 = []
for a, b in zip(x_personal_test_3, list(y_personal_test.values[:, 12:])):
    pred = model(torch.Tensor(np.expand_dims(np.expand_dims(a, 0), 0))).detach().numpy()
    print(pred, b)
    p_test_prediction_task3.append(pred)
p_test_prediction_task3 = np.concatenate(p_test_prediction_task3)
for col in range(4):
    p_test_prediction_df[train_labels.columns[12 + col]] = p_test_prediction_task3[:, col]

prediction = []
for a in x_test_scaled_task3:
    prediction.append(model(torch.Tensor(np.expand_dims(np.expand_dims(a, 0), 0))).detach().numpy())
prediction = np.concatenate(prediction)
for col in range(4):
    prediction_df[train_labels.columns[12 + col]] = prediction[:, col]

# In[12]:
#
#
# Task 3

# VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
#
# from sklearn.linear_model import LinearRegression
#
# for label in VITALS:
#     y_temp = train_labels[label]
#     reg = LinearRegression().fit(X_train_scaled_, y_temp)  # fitting the data
#
#     y_p_test_pred = pd.DataFrame(reg.predict(X_p_test_scaled_), columns=[label])
#     p_test_prediction_df = pd.concat([p_test_prediction_df, y_p_test_pred], axis = 1, sort= False)
#
#     y_pred = pd.DataFrame(reg.predict(X_test_scaled_), columns=[label])
#     prediction_df = pd.concat([prediction_df, y_pred], axis=1, sort=False)

# %% Get score
import score_submission

scores, score = score_submission.get_score(y_personal_test, p_test_prediction_df)
print('score: ', score)
# In[13]:

prediction_df.to_csv(r'predictions_.csv', index=False, float_format='%.3f')
prediction_df.to_csv('prediction_.zip', index=False, float_format='%.3f', compression='zip')

print('done')
