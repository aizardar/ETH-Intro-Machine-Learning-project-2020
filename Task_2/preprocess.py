import os
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import sklearn
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, ParameterGrid, GridSearchCV
from sklearn.metrics import roc_auc_score
from score_submission import get_score
import os
from utils import ExperimentLogger, mkdir, get_model
from sklearn.ensemble import RandomForestClassifier
import uuid


# In[ ]:

def impute_KNN(df):
    imputed_df = pd.DataFrame(columns=df.columns)
    imp = sklearn.impute.KNNImputer(n_neighbors=1)
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


def impute_mode(df):
    for column in df.columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    return df


def preprocess(task1_parameters, with_val_set, train_features, val_features, test, val_labels, train_labels):
    """
    The preprocessing (imputing with KNN and extracting the features) takes a lot of time, better to do it once and store the data.
    """
    if not os.path.exists('imputed_data/new_train_features_imp_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                                              task1_parameters[
                                                                                                  'drop_features'])):
        print("Imputing training data")

        # check if KNN imputed data exists
        if os.path.exists('imputed_data/knn_imputed_train_val-{}_featdrop-{}.csv'.format(with_val_set, task1_parameters[
            'drop_features'])):
            train_features_imp = pd.read_csv(
                'imputed_data/knn_imputed_train_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                               task1_parameters['drop_features']))
            train_features_imp = impute_mode(train_features_imp)

        else:
            train_features_imp = impute_KNN(train_features)
            train_features_imp.to_csv('imputed_data/knn_imputed_train_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                                                     task1_parameters[
                                                                                                         'drop_features']),
                                      index=False)
            train_features_imp = impute_mode(train_features_imp)
        if with_val_set:
            # check if KNN imputed val set exists
            if os.path.exists('imputed_data/knn_imputed_train_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                                             task1_parameters[
                                                                                                 'drop_features'])):

                # get validation features
                if os.path.exists('imputed_data/knn_imputed_val_featdrop-{}.csv'.format(task1_parameters[
                                                                                            'drop_features'])):
                    val_features_imp_ = pd.read_csv(
                        'imputed_data/knn_imputed_val_featdrop-{}.csv'.format(task1_parameters[
                                                                                  'drop_features']))
                else:
                    val_features_imp_ = impute_KNN(val_features)
                    val_features_imp_.to_csv('imputed_data/knn_imputed_val_featdrop-{}.csv'.format(task1_parameters[
                                                                                                       'drop_features']),
                                             index=False)

        print("Imputing test data")
        # check if KNN imputed test set exists
        if os.path.exists('imputed_data/knn_imputed_test_featdrop-{}.csv'.format(task1_parameters[
                                                                                     'drop_features'])):
            test_imputed = pd.read_csv('imputed_data/knn_imputed_test_featdrop-{}.csv'.format(task1_parameters[
                                                                                                  'drop_features']))
        else:
            test_imputed = impute_KNN(test)
            test_imputed.to_csv('imputed_data/knn_imputed_test_featdrop-{}.csv'.format(task1_parameters[
                                                                                           'drop_features']),
                                index=False)

        print("###############################")

        # In[ ]:

        if with_val_set:
            val_features_imp = impute_mode(val_features_imp_)
        test = impute_mode(test_imputed)

        # In[ ]:

        # Here, we would like to have one patient per row. Each feature has a time-series observations and we will extract standard set of features.

        def feature_extract(column, df):
            minimum = df[column].min()
            maximum = df[column].max()
            mean = np.float32(df[column].mean())
            standard_deviation = np.float32(df[column].std())
            median = np.float(df[column].median())
            skewness = np.float(df[column].skew())
            kurtosis = np.float(df[column].kurtosis())

            return minimum, maximum, mean, standard_deviation, median, skewness, kurtosis

        # Let's create a dictionary to store new features

        def new_df(df):
            new_df = {}

            for column in df.columns[(df.columns != 'pid') & (df.columns != 'Age') & (
                    df.columns != 'Time')]:  # We don't really need pid, Age and Time
                new_df[column + '_minimum'] = []
                new_df[column + '_maximum'] = []
                new_df[column + '_mean'] = []
                new_df[column + '_std'] = []
                new_df[column + '_median'] = []
                new_df[column + '_skewness'] = []
                new_df[column + '_kurtosis'] = []

            new_df['pid'] = []
            new_df['Age'] = []

            # Create a new dataframe with new extracted features

            for pid in tqdm(df.pid.unique()):

                for column in df.columns[(df.columns != 'pid') & (df.columns != 'Age') & (df.columns != 'Time')]:
                    # print(column)
                    minimum, maximum, mean, standard_deviation, median, skewness, kurtosis = feature_extract(column,
                                                                                                             df[
                                                                                                                 df.pid == pid])
                    # print(minimum, maximum, mean, standard_deviation)
                    new_df[column + '_minimum'].append(minimum)
                    new_df[column + '_maximum'].append(maximum)
                    new_df[column + '_mean'].append(mean)
                    new_df[column + '_std'].append(standard_deviation)
                    new_df[column + '_median'].append(median)
                    new_df[column + '_skewness'].append(skewness)
                    new_df[column + '_kurtosis'].append(kurtosis)

                new_df['pid'].append(pid)
                age = np.array(df.loc[df['pid'] == pid].Age)
                # print (age[0])
                new_df['Age'].append(age[0])

            new_df = pd.DataFrame.from_dict(new_df)
            return new_df

        print("Extracting new features from the training data")

        new_train_features_imp = new_df(train_features_imp)
        if with_val_set:
            new_val_features_imp = new_df(val_features_imp)
        print("###############################")

        print("Extracting new features from the test data")
        if not os.path.exists(
                'imputed_data/new_test_features_imp_featdrop-{}.csv'.format(task1_parameters['drop_features'])):
            new_test_features_imp = new_df(test)
            new_test_features_imp.to_csv(
                'imputed_data/new_test_features_imp_featdrop-{}.csv'.format(task1_parameters['drop_features']),
                index=False)
        else:
            new_test_features_imp = pd.read_csv(
                'imputed_data/new_test_features_imp_featdrop-{}.csv'.format(task1_parameters['drop_features']))

        new_train_features_imp.to_csv('imputed_data/new_train_features_imp_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                                                          task1_parameters[
                                                                                                              'drop_features']),
                                      index=False)
        train_labels.to_csv(
            'imputed_data/train_labels_val-{}_featdrop-{}.csv'.format(with_val_set, task1_parameters['drop_features']),
            index=False)
        if with_val_set:
            new_val_features_imp.to_csv(
                'imputed_data/new_val_features_imp_featdrop-{}.csv'.format(task1_parameters['drop_features']),
                index=False)
            val_labels.to_csv('imputed_data/val_labels_featdrop-{}.csv'.format(task1_parameters['drop_features']),
                              index=False)

    else:
        print("Loading training data")
        new_train_features_imp = pd.read_csv(
            'imputed_data/new_train_features_imp_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                                task1_parameters['drop_features']))
        new_val_features_imp = pd.read_csv(
            'imputed_data/new_val_features_imp_featdrop-{}.csv'.format(task1_parameters['drop_features']))
        new_test_features_imp = pd.read_csv(
            'imputed_data/new_test_features_imp_featdrop-{}.csv'.format(task1_parameters['drop_features']))
        train_labels = pd.read_csv(
            'imputed_data/train_labels_val-{}_featdrop-{}.csv'.format(with_val_set, task1_parameters['drop_features']))
        val_labels = pd.read_csv('imputed_data/val_labels_featdrop-{}.csv'.format(task1_parameters['drop_features']))

    return val_labels, train_labels, new_test_features_imp, new_val_features_imp, new_train_features_imp
