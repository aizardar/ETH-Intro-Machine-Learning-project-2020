import torch
import pandas as pd
import sklearn
import numpy as np
import tqdm
import os
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping

class testDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
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

class MyNetwork(torch.nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size= (2,1), stride= 1, padding= 0),
            torch.nn.ReLU(inplace= True),
            torch.nn.MaxPool2d((2,1)),
            Flatten(),
            torch.nn.Linear(11200, 50),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(50, 4)
        )

    def forward(self, x):
        return self.network(x)

class Net(LightningModule):
    def __init__(self, train_dataset, val_dataset):
        super(Net, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size= (2,1), stride= 1, padding= 0),
            torch.nn.ReLU(inplace= True),
            torch.nn.MaxPool2d((2,1)),
            Flatten(),
            torch.nn.Linear(11200, 50),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(50, 4)
        )

    def forward(self, x):
        return self.network(x)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size= 2048)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size= 2048)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr= 1e-4)

    def training_step(self, batch, batch_index):
        data, target = batch
        output = self.forward(data)
        loss = self.mse(output, target)
        return {'loss': loss}

    def validation_step(self, batch, batch_index):
        data, target = batch
        output = self.forward(data)
        loss = self.mse(output, target)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def mse(self, logits, labels):
        return torch.nn.functional.mse_loss(logits, labels)


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

if __name__ == '__main__':

    SEED = 42

    train_features = pd.read_csv("train_features.csv").sort_values(by='pid')
    test = pd.read_csv("test_features.csv")
    train_labels = pd.read_csv("train_labels.csv").sort_values(by='pid')

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

    feature_cols = train_features.columns.values[
        (train_features.columns.values != 'pid') & (train_features.columns.values != 'Time')]

    train_features_ = impute_mode(train_features_imp)
    test_ = impute_mode(test_imputed)


    train_labels = pd.read_csv("train_labels.csv").sort_values(by='pid')
    train_labels_task3 = list(train_labels.values[:, 12:])

    x_train = []
    for i, subject in enumerate(list(dict.fromkeys(train_features_['pid'].values.tolist()))):
        x_train.append(train_features_.loc[train_features_['pid'] == subject, feature_cols].values)

    x_test_scaled_task3 = []
    for i, subject in enumerate(list(dict.fromkeys(test_['pid'].values.tolist()))):
        x_test_scaled_task3.append(test_.loc[test_['pid'] == subject, feature_cols].values)

    train_features, x_personal_test, train_labels, y_personal_test = train_test_split(x_train, train_labels_task3,test_size=0.2, random_state=SEED)

    scaler = sklearn.preprocessing.RobustScaler()

    x_temp = np.concatenate(train_features)
    x_temp = scaler.fit_transform(x_temp)
    x_train = np.vsplit(x_temp, x_temp.shape[0] / 12)

    x_temp = np.concatenate(x_personal_test)
    x_temp = scaler.transform(x_temp)
    x_personal_test = np.vsplit(x_temp, x_temp.shape[0] / 12)

    x_temp = np.concatenate(x_test_scaled_task3)
    x_temp = scaler.transform(x_temp)
    x_test_scaled_task3 = np.vsplit(x_temp, x_temp.shape[0] / 12)

    input_shape = x_train[0].shape

    x_train, x_val, y_train, y_val = train_test_split(x_train, train_labels, test_size=0.2, random_state=SEED)

    train_dataset = MyDataset(x_train, y_train)
    val_dataset = MyDataset(x_val, y_val)
    personal_test_dataset = MyDataset(x_test_scaled_task3, y_personal_test)
    model = Net(train_dataset, val_dataset)
    early_stopping = EarlyStopping('val_loss')
    trainer = Trainer(early_stop_callback=early_stopping)
    trainer.fit(model)
    model.forward(personal_test_dataset)
