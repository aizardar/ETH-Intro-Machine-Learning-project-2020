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

train_features = train_features_.groupby('pid').mean().reset_index()
test = test_.groupby('pid').mean().reset_index()
# %% split into personal test set and train set
train_features, x_personal_test, train_labels, y_personal_test = train_test_split(train_features, train_labels,
                                                                                  test_size=0.2, random_state=SEED)

p_test_prediction_df = pd.DataFrame(y_personal_test['pid'].values, columns=['pid'])

from sklearn.preprocessing import RobustScaler

feature_cols = train_features.columns.values[
    (train_features.columns.values != 'pid') & (train_features.columns.values != 'Time')]
label_cols1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
               'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
               'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis']

x_train1 = []
for i, subject in enumerate(list(dict.fromkeys(train_features['pid'].values.tolist()))):
    x_train1.append(train_features_.loc[train_features_['pid'] == subject, feature_cols].values)

x_personal_test_1 = []
for i, subject in enumerate(list(dict.fromkeys(x_personal_test['pid'].values.tolist()))):
    x_personal_test_1.append(train_features_.loc[train_features_['pid'] == subject, feature_cols].values)

x_test_scaled_task1 = []
for i, subject in enumerate(list(dict.fromkeys(test_['pid'].values.tolist()))):
    x_test_scaled_task1.append(test_.loc[test_['pid'] == subject, feature_cols].values)

scaler = RobustScaler()

x_temp = np.concatenate(x_train1)
x_temp = scaler.fit_transform(x_temp)
x_train = np.vsplit(x_temp, x_temp.shape[0] / 12)

x_temp = np.concatenate(x_personal_test_1)
x_temp = scaler.transform(x_temp)
x_personal_test_1 = np.vsplit(x_temp, x_temp.shape[0] / 12)

x_temp = np.concatenate(x_test_scaled_task1)
x_temp = scaler.transform(x_temp)
x_test_scaled_task1 = np.vsplit(x_temp, x_temp.shape[0] / 12)

input_shape = x_train[0].shape

# %%
from ignite.contrib.metrics.regression import R2Score
from ignite.contrib.metrics import ROC_AUC
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping
from pytorch_lightning.loggers import TensorBoardLogger


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


class Net1(LightningModule):
    def __init__(self, train_dataset, val_dataset):
        super(Net1, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        kernel_size1 = 2
        kernel_size2 = 1
        num_params = ((12 - kernel_size1 + 1) // 2) * ((35 - kernel_size2 + 1) // 1) * 64
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(kernel_size1, kernel_size2), stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d((2, 1)),
            Flatten(),
            torch.nn.Linear(num_params, 50),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(50, 11),
            torch.nn.Sigmoid()
        )

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
        loss = self.dice_loss(output, target)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_index):
        data, target = batch
        output = self.forward(data)
        loss = self.dice_loss(output, target)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def dice_loss(self, input, target):
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))


# %%
x_train, x_val, y_train, y_val = train_test_split(x_train, train_labels[label_cols1].values, test_size=0.2,
                                                  random_state=SEED)

train_dataset = MyDataset(x_train, y_train)
val_dataset = MyDataset(x_val, y_val)
logger = TensorBoardLogger("tb_logs", name="task1")
checkpoint_callback = ModelCheckpoint(filepath='temp/', verbose=True)
early_stopper = early_stopping.EarlyStopping(monitor='val_loss', verbose=True, patience=8 * 5)
model = Net1(train_dataset, val_dataset)
trainer = Trainer(checkpoint_callback=checkpoint_callback, profiler=True, logger=logger)
trainer.fit(model)

p_test_prediction_task1 = []
for a, b in zip(x_personal_test_1, list(y_personal_test[label_cols1].values)):
    pred = model(torch.Tensor(np.expand_dims(np.expand_dims(a, 0), 0))).detach().numpy()
    # print(pred, b)
    p_test_prediction_task1.append(pred)
p_test_prediction_task1 = np.concatenate(p_test_prediction_task1)
p_test_prediction_df = pd.DataFrame(p_test_prediction_task1, columns=label_cols1)
p_test_prediction_df['pid'] = y_personal_test['pid'].values

prediction = []
for a in x_test_scaled_task1:
    prediction.append(model(torch.Tensor(np.expand_dims(np.expand_dims(a, 0), 0))).detach().numpy())
prediction = np.concatenate(prediction)
prediction_df = pd.DataFrame(prediction, columns=label_cols1)
prediction_df['pid'] = test.pid.values

# In[ ]:
#
#
# %%
from ignite.contrib.metrics.regression import R2Score
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping
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
    def __init__(self, train_dataset, val_dataset):
        super(Net, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        kernel_size1 = 2
        kernel_size2 = 1
        num_params = ((12 - kernel_size1 + 1) // 2) * ((35 - kernel_size2 + 1) // 1) * 64
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(kernel_size1, kernel_size2), stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d((2, 1)),
            Flatten(),
            torch.nn.Linear(num_params, 50),
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
        self.val_r2_score = avg_r2
        tensorboard_logs = {'val_loss': avg_loss, 'val_r2_score1': avg_r2_score1, 'val_r2_score2': avg_r2_score2,
                            'val_r2_score3': avg_r2_score3, 'val_r2_score4': avg_r2_score4, 'avg_r2': avg_r2, }
        print(avg_r2)
        return {'val_loss': avg_loss, 'val_r2_score': avg_r2, 'log': tensorboard_logs}

    def mse(self, logits, labels):
        return torch.nn.functional.mse_loss(logits, labels)


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
early_stopper = early_stopping.EarlyStopping(monitor='val_r2_score', verbose=True, patience=8 * 5)
model = Net(train_dataset, val_dataset)
trainer = Trainer(checkpoint_callback=checkpoint_callback, logger=logger, profiler=True)
trainer.fit(model)

p_test_prediction_task3 = []
for a, b in zip(x_personal_test_3, list(y_personal_test.values[:, 12:])):
    pred = model(torch.Tensor(np.expand_dims(np.expand_dims(a, 0), 0))).detach().numpy()
    # print(pred, b)
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

# %% Get score
import score_submission

scores, score = score_submission.get_score(y_personal_test, p_test_prediction_df)
print('score: ', score)
# In[13]:

prediction_df.to_csv(r'predictions_.csv', index=False, float_format='%.3f')
prediction_df.to_csv('prediction_.zip', index=False, float_format='%.3f', compression='zip')

print('done')
