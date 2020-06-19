import os
from sklearn.metrics import roc_auc_score, r2_score
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from models import dice_coef_loss
from score_submission import get_score
from tqdm import tqdm
from utils import handle_nans, scaling
from utils import train_model


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


def preprocess(parameters, with_val_set, train_features, val_features, test, val_labels, train_labels):
    """
    The preprocessing (imputing with KNN and extracting the features) takes a lot of time, better to do it once and store the data.
    """

    if not os.path.exists('imputed_data/new_train_features_imp_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                                              parameters[
                                                                                                  'drop_features'])):
        print("Imputing training data")

        # check if KNN imputed data exists
        if os.path.exists('imputed_data/knn_imputed_train_val-{}_featdrop-{}.csv'.format(with_val_set, parameters[
            'drop_features'])):
            train_features_imp = pd.read_csv(
                'imputed_data/knn_imputed_train_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                               parameters['drop_features']))
            train_features_imp = impute_mode(train_features_imp)

        else:
            train_features_imp = impute_KNN(train_features)
            train_features_imp.to_csv('imputed_data/knn_imputed_train_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                                                     parameters[
                                                                                                         'drop_features']),
                                      index=False)
            train_features_imp = impute_mode(train_features_imp)
        if with_val_set:
            # check if KNN imputed val set exists
            if os.path.exists('imputed_data/knn_imputed_train_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                                             parameters[
                                                                                                 'drop_features'])):

                # get validation features
                if os.path.exists('imputed_data/knn_imputed_val_featdrop-{}.csv'.format(parameters[
                                                                                            'drop_features'])):
                    val_features_imp_ = pd.read_csv(
                        'imputed_data/knn_imputed_val_featdrop-{}.csv'.format(parameters[
                                                                                  'drop_features']))
                else:
                    val_features_imp_ = impute_KNN(val_features)
                    val_features_imp_.to_csv('imputed_data/knn_imputed_val_featdrop-{}.csv'.format(parameters[
                                                                                                       'drop_features']),
                                             index=False)

        print("Imputing test data")
        # check if KNN imputed test set exists
        if os.path.exists('imputed_data/knn_imputed_test_featdrop-{}.csv'.format(parameters[
                                                                                     'drop_features'])):
            test_imputed = pd.read_csv('imputed_data/knn_imputed_test_featdrop-{}.csv'.format(parameters[
                                                                                                  'drop_features']))
        else:
            test_imputed = impute_KNN(test)
            test_imputed.to_csv('imputed_data/knn_imputed_test_featdrop-{}.csv'.format(parameters[
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
                'imputed_data/new_test_features_imp_featdrop-{}.csv'.format(parameters['drop_features'])):
            new_test_features_imp = new_df(test)
            new_test_features_imp.to_csv(
                'imputed_data/new_test_features_imp_featdrop-{}.csv'.format(parameters['drop_features']),
                index=False)
        else:
            new_test_features_imp = pd.read_csv(
                'imputed_data/new_test_features_imp_featdrop-{}.csv'.format(parameters['drop_features']))

        new_train_features_imp.to_csv('imputed_data/new_train_features_imp_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                                                          parameters[
                                                                                                              'drop_features']),
                                      index=False)
        train_labels.to_csv(
            'imputed_data/train_labels_val-{}_featdrop-{}.csv'.format(with_val_set, parameters['drop_features']),
            index=False)
        if with_val_set:
            new_val_features_imp.to_csv(
                'imputed_data/new_val_features_imp_featdrop-{}.csv'.format(parameters['drop_features']),
                index=False)
            val_labels.to_csv('imputed_data/val_labels_featdrop-{}.csv'.format(parameters['drop_features']),
                              index=False)

    else:
        print("Loading training data")
        new_train_features_imp = pd.read_csv(
            'imputed_data/new_train_features_imp_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                                parameters['drop_features']))
        new_val_features_imp = pd.read_csv(
            'imputed_data/new_val_features_imp_featdrop-{}.csv'.format(parameters['drop_features']))
        new_test_features_imp = pd.read_csv(
            'imputed_data/new_test_features_imp_featdrop-{}.csv'.format(parameters['drop_features']))
        train_labels = pd.read_csv(
            'imputed_data/train_labels_val-{}_featdrop-{}.csv'.format(with_val_set, parameters['drop_features']))
        val_labels = pd.read_csv('imputed_data/val_labels_featdrop-{}.csv'.format(parameters['drop_features']))

    return val_labels, train_labels, new_test_features_imp, new_val_features_imp, new_train_features_imp


def NN_pipeline(parameters, X_train_df, y_train_df, X_test_df, X_val_df, y_val_df, task,
                experiment_logger, with_val_set):
    if not os.path.exists('predictions.csv'):
        # create submission dataframes that will be filled with the predictions
        df_submission = pd.DataFrame(X_test_df.pid.values, columns=['pid'])
        new_df_submission = True
    else:
        df_submission = pd.read_csv('predictions.csv')
        new_df_submission = False

    if new_df_submission or not os.path.exists('val_predictions.csv'):
        df_submission_val = pd.DataFrame(X_val_df.pid.values, columns=['pid'])
    else:
        df_submission_val = pd.read_csv('val_predictions.csv')

    print("Imputing training data")

    # check if KNN imputed data exists
    if os.path.exists('imputed_data/knn_imputed_train_val-{}_featdrop-{}.csv'.format(with_val_set, parameters[
        'drop_features'])):
        train_features_imp = pd.read_csv(
            'imputed_data/knn_imputed_train_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                           parameters['drop_features']))
        train_features_imp = impute_mode(train_features_imp)

    else:
        train_features_imp = impute_KNN(X_train_df)
        train_features_imp.to_csv('imputed_data/knn_imputed_train_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                                                 parameters[
                                                                                                     'drop_features']),
                                  index=False)
        train_features_imp = impute_mode(train_features_imp)
    if with_val_set:
        # check if KNN imputed val set exists
        if os.path.exists('imputed_data/knn_imputed_train_val-{}_featdrop-{}.csv'.format(with_val_set,
                                                                                         parameters[
                                                                                             'drop_features'])):

            # get validation features
            if os.path.exists('imputed_data/knn_imputed_val_featdrop-{}.csv'.format(parameters[
                                                                                        'drop_features'])):
                val_features_imp_ = pd.read_csv(
                    'imputed_data/knn_imputed_val_featdrop-{}.csv'.format(parameters[
                                                                              'drop_features']))
            else:
                val_features_imp_ = impute_KNN(X_val_df)
                val_features_imp_.to_csv('imputed_data/knn_imputed_val_featdrop-{}.csv'.format(parameters[
                                                                                                   'drop_features']),
                                         index=False)

    print("Imputing test data")
    # check if KNN imputed test set exists
    if os.path.exists('imputed_data/knn_imputed_test_featdrop-{}.csv'.format(parameters[
                                                                                 'drop_features'])):
        test_imputed = pd.read_csv('imputed_data/knn_imputed_test_featdrop-{}.csv'.format(parameters[
                                                                                              'drop_features']))
    else:
        test_imputed = impute_KNN(X_test_df)
        test_imputed.to_csv('imputed_data/knn_imputed_test_featdrop-{}.csv'.format(parameters[
                                                                                       'drop_features']),
                            index=False)

    train_path = 'nan_handling/train{}_{}.csv'.format(parameters['nan_handling'], parameters['drop_features'])
    x_test_path = 'nan_handling/test{}_{}.csv'.format(parameters['nan_handling'], parameters['drop_features'])
    x_val_path = 'nan_handling/val{}_{}.csv'.format(parameters['nan_handling'], parameters['drop_features'])
    y_val_df = y_val_df.reset_index(drop=True)
    if os.path.isfile(train_path) and os.path.isfile(x_val_path) and os.path.isfile(x_test_path):
        X_train_df = pd.read_csv(train_path)
        X_test_df = pd.read_csv(x_test_path)
        X_val_df = pd.read_csv(x_val_path)
    else:
        X_train_df, imp = handle_nans(train_features_imp, parameters, 42)
        X_train_df.to_csv(train_path, index=False)
        if not parameters['nan_handling'] in ['zero', 'minusone']:
            X_test_df = pd.DataFrame(data=imp.transform(test_imputed), columns=test_imputed.columns)
            X_val_df = pd.DataFrame(data=imp.transform(val_features_imp_), columns=val_features_imp_.columns)
        else:
            X_test_df, _ = handle_nans(test_imputed, parameters, 42)
            X_val_df, _ = handle_nans(val_features_imp_, parameters, 42)
        X_test_df.to_csv(x_test_path, index=False)
        X_val_df.to_csv(x_val_path, index=False)

    if parameters['collapse_time'] == 'yes':
        x_train = X_train_df.groupby('pid').mean().reset_index()
        x_test = X_test_df.groupby('pid').mean().reset_index()
        x_val = X_val_df.groupby('pid').mean().reset_index()
        input_shape = x_train.shape

    if parameters['collapse_time'] == 'no':
        x_train = []
        for i, subject in enumerate(list(dict.fromkeys(y_train_df['pid'].values.tolist()))):
            if parameters['with_time'] == 'yes':
                x_train.append(X_train_df.loc[X_train_df['pid'] == subject].values[:, 1:])
            elif parameters['with_time'] == 'no':
                x_train.append(X_train_df.loc[X_train_df['pid'] == subject].values[:, 2:])

        x_val = []
        for i, subject in enumerate(list(dict.fromkeys(y_val_df['pid'].values.tolist()))):
            if parameters['with_time'] == 'yes':
                x_val.append(X_val_df.loc[X_val_df['pid'] == subject].values[:, 1:])
            elif parameters['with_time'] == 'no':
                x_val.append(X_val_df.loc[X_val_df['pid'] == subject].values[:, 2:])

        x_test = []
        for i, subject in enumerate(list(dict.fromkeys(X_test_df['pid'].values.tolist()))):
            if parameters['with_time'] == 'yes':
                x_test.append(X_test_df.loc[X_test_df['pid'] == subject].values[:, 1:])
            if parameters['with_time'] == 'no':
                x_test.append(X_test_df.loc[X_test_df['pid'] == subject].values[:, 2:])

        input_shape = x_train[0].shape

    if not parameters['standardizer'] == 'none':
        x_train, scaler = scaling(x_train, parameters)
        x_val, _ = scaling(x_val, parameters, scaler)
        x_test, _ = scaling(x_test, parameters, scaler)
    else:
        x_scaler = None

    if parameters['collapse_time'] == 'yes':
        if parameters['task'] == 12:
            y_train1 = y_train_df.iloc[:, 1:12]
        else:
            y_train1 = y_train_df.iloc[:, 1:11]
        y_train2 = y_train_df.iloc[:, 11]
        y_train3 = y_train_df.iloc[:, 12:]
    else:
        if parameters['task'] == 12:
            y_train1 = list(y_train_df.values[:, 1:12])
        else:
            y_train1 = list(y_train_df.values[:, 1:11])
        y_train2 = list(y_train_df.values[:, 11])
        y_train3 = list(y_train_df.values[:, 12:])

    if task == 1 or task == 12:
        print('\n\n**** Training model1 **** \n\n')
        loss = parameters['task1_loss']
        if loss == 'dice':
            loss = dice_coef_loss
        model1 = train_model(parameters, input_shape, x_train, y_train1, loss, parameters['epochs'], 42,
                             parameters['task'],
                             y_train_df)

    if task == 2:
        loss = parameters['task2_loss']
        if loss == 'dice':
            loss = dice_coef_loss
        print('\n\n**** Training model2 **** \n\n')
        model2 = train_model(parameters, input_shape, x_train, y_train2, loss, parameters['epochs'], 42, 2,
                             y_train_df)

    if task == 3:
        loss = parameters['task3_loss']
        if loss == 'dice':
            loss = dice_coef_loss
        elif loss == 'huber':
            loss = 'huber_loss'
        print('\n\n**** Training model3 **** \n\n')
        model3 = train_model(parameters, input_shape, x_train, y_train3, loss, parameters['epochs'], 42, 3,
                             y_train_df)

    if parameters['model'] == 'resnet' or parameters['model'] == 'simple_conv_model':
        x_test = np.expand_dims(x_test, -1)
        x_val = np.expand_dims(x_val, -1)

    if not parameters['model'].startswith('lin'):
        test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
        test_dataset = test_dataset.batch(batch_size=parameters['batch_size'])
        # test_dataset = test_dataset.batch()


        val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
        val_dataset = val_dataset.batch(batch_size=parameters['batch_size'])

    if not os.path.exists('test_pred.csv'):
        test_df = pd.DataFrame(columns=sample.columns)
        test_df['pid'] = y_val_df['pid'].values
        test_df.to_csv('test_pred.csv')
    else:
        test_df = pd.read_csv('test_pred.csv')

    if task == 1:
        # predict on validation set
        val_prediction1 = model1.predict(val_dataset)
        val_prediction_df = pd.DataFrame(val_prediction1, columns=y_val_df.columns[1:11])
        val_score = get_score(y_val_df, val_prediction_df, task)
        # if experiment_logger.previous_results_empty or val_score > experiment_logger.previous_results[
        #     'task1_score'].max():
        #     for col in val_prediction_df.columns:
        #         val_df[col] = val_prediction_df[col]
        #     val_df.reindex(columns=sample.columns)

    if task == 12:
        val_prediction12 = model1.predict(val_dataset)
        val_prediction_df = pd.DataFrame(val_prediction12, columns=y_val_df.columns[1:12])
        predictions_test = model1.predict(test_dataset)
        test_prediction_df = pd.DataFrame(predictions_test, columns=y_val_df.columns[1:12])
        for label in y_val_df.columns[1:12]:
            val_score = roc_auc_score(y_val_df[label], val_prediction_df[label])
            print(f'task 12 score label {label}: ', val_score)
            experiment_logger.df['{}_score'.format(label)] = val_score
            if label == 'LABEL_Sepsis':
                if experiment_logger.previous_results_empty or (val_score > experiment_logger.previous_results2[
                '{}_score'.format(label)].max() and val_score > experiment_logger.previous_results12[
                '{}_score'.format(label)].max()):
                    if label in df_submission_val.columns:
                        df_submission_val.drop(label, axis=1, inplace=True)
                    if label in df_submission.columns:
                        df_submission.drop(label, axis=1, inplace=True)
                    df_submission_val = pd.concat([df_submission_val, val_prediction_df[label]], axis=1, sort=False)
                    df_submission = pd.concat([df_submission, test_prediction_df[label]], axis=1, sort=False)
            else:
                if experiment_logger.previous_results_empty or (val_score > experiment_logger.previous_results1[
                '{}_score'.format(label)].max() and val_score > experiment_logger.previous_results12[
                '{}_score'.format(label)].max()):
                    if label in df_submission_val.columns:
                        df_submission_val.drop(label, axis=1, inplace=True)
                    if label in df_submission.columns:
                        df_submission.drop(label, axis=1, inplace=True)
                    df_submission_val = pd.concat([df_submission_val, val_prediction_df[label]], axis=1, sort=False)
                    df_submission = pd.concat([df_submission, test_prediction_df[label]], axis=1, sort=False)

    if task == 2:
        val_prediction2 = model2.predict(val_dataset)
        val_prediction_df = pd.DataFrame(val_prediction2, columns=[y_val_df.columns[11]])
        # val_prediction_df['pid'] = y_val_df['pid']
        label = 'LABEL_Sepsis'
        val_score = roc_auc_score(y_val_df[label], val_prediction_df[label])
        print('task 2 score: ', val_score)
        experiment_logger.df['{}_score'.format('LABEL_Sepsis')] = val_score
        if experiment_logger.previous_results_empty or val_score > experiment_logger.previous_results[
            'task2_score'].max():
            predictions_test = model2.predict(test_dataset)
            predict_prob_val = pd.DataFrame(np.ravel(val_prediction2[:, 1]), columns=['LABEL_Sepsis'])
            predict_prob_test = pd.DataFrame(np.ravel(predictions_test[:, 1]), columns=['LABEL_Sepsis'])
            if 'LABEL_Sepsis' in df_submission_val.columns:
                df_submission_val.drop('LABEL_Sepsis', axis=1, inplace=True)
            if 'LABEL_Sepsis' in df_submission.columns:
                df_submission.drop('LABEL_Sepsis', axis=1, inplace=True)
            df_submission_val = pd.concat([df_submission_val, predict_prob_val], axis=1, sort=False)
            df_submission = pd.concat([df_submission, predict_prob_test], axis=1, sort=False)

    if task == 3:
        VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
        val_prediction3 = model3.predict(val_dataset)
        val_prediction_df = pd.DataFrame(val_prediction3, columns=y_val_df.columns[12:])
        predictions_test = model3.predict(test_dataset)
        test_prediction_df = pd.DataFrame(predictions_test, columns=y_val_df.columns[12:])
        for label in VITALS:
            val_score = r2_score(val_prediction_df[label], y_val_df[label])
            print(f'task 3 score label {label}: ', val_score)
            experiment_logger.df['{}_score'.format(label)] = val_score
            if experiment_logger.previous_results_empty or val_score > experiment_logger.previous_results[
                '{}_score'.format(label)].max():
                if label in df_submission_val.columns:
                    df_submission_val.drop(label, axis=1, inplace=True)
                if label in df_submission.columns:
                    df_submission.drop(label, axis=1, inplace=True)
                df_submission_val = pd.concat([df_submission_val, val_prediction_df[label]], axis=1, sort=False)
                df_submission = pd.concat([df_submission, test_prediction_df[label]], axis=1, sort=False)

    return df_submission_val, df_submission
