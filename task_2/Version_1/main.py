#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import sklearn
from tqdm import tqdm


# In[2]:


train_features = pd.read_csv("train_features.csv").sort_values(by='pid')
test = pd.read_csv("test_features.csv")
train_labels = pd.read_csv("train_labels.csv").sort_values(by='pid')


# In[3]:


# Let's get rid of Hgb, HCO3, ABPd, and Bilirubin_direct from the training and test data !

train_features.drop('Bilirubin_direct', axis=1, inplace=True)
train_features.drop('HCO3', axis=1, inplace=True)
train_features.drop('Hgb', axis=1, inplace=True)
train_features.drop('ABPd', axis=1, inplace=True)

test.drop('Bilirubin_direct', axis=1, inplace=True)
test.drop('HCO3', axis=1, inplace=True)
test.drop('Hgb', axis=1, inplace=True)
test.drop('ABPd', axis=1, inplace=True) 


# In[4]:


def impute_KNN(df):
    imputed_df = pd.DataFrame(columns=df.columns)
    imp = sklearn.impute.KNNImputer(n_neighbors=2)
    for pid in tqdm(np.unique(df['pid'].values)):
        temp_df = df.loc[df['pid'] == pid]
        temp_df2 = temp_df.dropna(axis = 'columns', how = 'all')
        imp.fit(temp_df2)
        temp_df2 = pd.DataFrame(data = imp.transform(temp_df2), columns = temp_df2.columns)
        for key in temp_df.columns:
            if temp_df[key].isna().all():
                temp_df2[key] = np.nan
        imputed_df = imputed_df.append(temp_df2, sort = True)
    imputed_df.reindex(columns = df.columns)
    return imputed_df


# In[5]:


train_features_imp = impute_KNN(train_features)
test_imputed = impute_KNN(test)


# In[6]:


def impute_mode(df):
    for column in df.columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    return df


# In[7]:


train_features = impute_mode(train_features_imp)
test = impute_mode(test_imputed)


# In[8]:


# Let's now take the mean of all pids so that we have just one row per patient 

train_features = train_features.groupby('pid').mean().reset_index()
test = test.groupby('pid').mean().reset_index()


# In[9]:


# Normalize the input features using the sklearn StandardScaler. This will set the mean to 0 and standard deviation to 1.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_train_scaled = pd.DataFrame(X_train_scaled, columns = train_features.columns)

X_test_scaled = scaler.transform(test)
X_test_scaled = pd.DataFrame(X_test_scaled,columns = test.columns)


# In[10]:


feature_cols = X_train_scaled.columns.values[(X_train_scaled.columns.values != 'pid') & (X_train_scaled.columns.values != 'Time')]

X_train_scaled = X_train_scaled[feature_cols]
X_test_scaled = X_test_scaled[feature_cols]


# In[11]:


# Task 1 and Task 2 

from sklearn.svm import SVC


TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
          'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
          'LABEL_Bilirubin_direct', 'LABEL_EtCO2','LABEL_Sepsis']


df = pd.DataFrame(test.pid.values, columns = ['pid'])


for label in tqdm(TESTS):
    

    if label == 'LABEL_BaseExcess':
        
        y_train_temp = train_labels[label]
        model = SVC(kernel = 'rbf', gamma = 0.01, C = 10, random_state=42, probability=True)
        model.fit(X_train_scaled, y_train_temp)
        predictions_prob_test = model.predict_proba(X_test_scaled)
        predict_prob_test = pd.DataFrame(np.ravel(predictions_prob_test[:,1]), columns = [label])
        df = pd.concat([df, predict_prob_test], axis = 1, sort = False)
        
    elif label == 'LABEL_TroponinI':

        y_train_temp = train_labels[label]
        model = SVC(kernel = 'rbf', gamma = 0.001, C = 1, random_state=42, probability=True)
        model.fit(X_train_scaled, y_train_temp)
        predictions_prob_test = model.predict_proba(X_test_scaled)
        predict_prob_test = pd.DataFrame(np.ravel(predictions_prob_test[:,1]), columns = [label])
        df = pd.concat([df, predict_prob_test], axis = 1, sort = False)
        
    elif label == 'LABEL_AST':
        
        y_train_temp = train_labels[label]
        model = SVC(kernel = 'rbf', gamma = 0.001, C = 1, random_state=42, probability=True)
        model.fit(X_train_scaled, y_train_temp)
        predictions_prob_test = model.predict_proba(X_test_scaled)
        predict_prob_test = pd.DataFrame(np.ravel(predictions_prob_test[:,1]), columns = [label])
        df = pd.concat([df, predict_prob_test], axis = 1, sort = False)
 
    else:
        
        y_train_temp = train_labels[label]
        model = SVC(kernel = 'rbf', gamma = 0.1, C = 1, random_state=42, probability=True)
        model.fit(X_train_scaled, y_train_temp)
        predictions_prob_test = model.predict_proba(X_test_scaled)
        predict_prob_test = pd.DataFrame(np.ravel(predictions_prob_test[:,1]), columns = [label])
        df = pd.concat([df, predict_prob_test], axis = 1, sort = False)


# In[ ]:





# In[12]:


# Task 3

VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

from sklearn.linear_model import LinearRegression

for label in VITALS:

    y_temp = train_labels[label]
    reg = LinearRegression().fit(X_train_scaled, y_temp) # fitting the data
    y_pred = pd.DataFrame(reg.predict(X_test_scaled), columns = [label])
    df = pd.concat([df, y_pred], axis = 1, sort = False)


# In[13]:


df.to_csv(r'predictions.csv', index = False, float_format='%.3f')

