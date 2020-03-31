import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn import preprocessing
import numpy as np

def handle_nans(X_train_df, params, seed):
    if params['nan_handling'] == 'impute':
        imp = IterativeImputer(max_iter=10, random_state=seed)
        imp.fit(X_train_df)
        X_train_df = pd.DataFrame(data = imp.transform(X_train_df), columns = X_train_df.columns)
        # X_train_df.to_csv('temp/imputed_taining_data.csv')
    elif params['nan_handling'] == 'knn':
        imp = KNNImputer(n_neighbors=2, weights="uniform")
        imp.fit(X_train_df)
        X_train_df = pd.DataFrame(data = imp.transform(X_train_df), columns = X_train_df.columns)
    elif params['nan_handling'] == 'minusone':
        X_train_df = X_train_df.fillna(-1)
    elif params['nan_handling'] == 'zero':
        X_train_df = X_train_df.fillna(0)
    elif params['nan_handling'] == 'drop':
        x_train_df = X_train_df.dropna(axis = 0, how = 'any')
    else:
        imp = SimpleImputer(missing_values=np.nan, strategy=params['nan_handling'])
        imp.fit(X_train_df)
        X_train_df = pd.DataFrame(data = imp.transform(X_train_df), columns = X_train_df.columns)
    return X_train_df

def scaler(params):
    if params['standardizer'] == 'RobustScaler':
        scaler = preprocessing.RobustScaler()
    elif params['standardizer'] == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    elif params['standardizer'] == 'maxabsscaler':
        scaler = preprocessing.MaxAbsScaler()
    elif params['standardizer'] == 'standardscaler':
        scaler = preprocessing.StandardScaler()
    return scaler