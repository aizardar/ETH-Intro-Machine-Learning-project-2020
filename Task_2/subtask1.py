import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import Huber
import models
from models import simple_model, threelayers
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn import preprocessing, svm
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import os
import utils
import functools, operator
from models import dice_coef_loss, build_model
from tqdm import tqdm
import random
from scipy.spatial.distance import dice
from score_submission import get_score

"""
Predict whether medical tests are ordered by a clinician in the remainder of the hospital stay: 0 means that there will be no further tests of this kind ordered, 1 means that at least one of a test of that kind will be ordered. In the submission file, you are asked to submit predictions in the interval [0, 1], i.e., the predictions are not restricted to binary. 0.0 indicates you are certain this test will not be ordered, 1.0 indicates you are sure it will be ordered. The corresponding columns containing the binary groundtruth in train_labels.csv are: LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2.
10 labels for this subtask

fill base excess with 0
fibrogen with -1

Questions:
    - include time axis in training data?
    - use a SVM for each value to predict?
"""
test = False
seed = 1
num_subjects = -1  # number of subjects out of 18995
epochs = 1000
TRIALS = 50
# Todo collapse persons over time to find outlayers?

search_space_dict_task1 = {
    'nan_handling': ['minusone'],
    'standardizer': ['RobustScaler'],
    'model': ['threelayers'],
    'batch_size': [2048],
    'impute_nn': ['yes'],
    'epochs': [epochs],
    'numbr_subjects': [num_subjects],
    'keras_tuner': ['no'],
    'task': [1],
    'task1_activation': ['sigmoid'],
    'task1_loss': ['dice'],
    'with_time': ['no'],
    'collapse_time': ['yes'],
}
search_space_dict_task2 = {
    # 'loss': ['dice','binary_crossentropy'],
    'nan_handling': ['minusone'],
    'standardizer': ['minmax'],
    # 'output_layer': ['sigmoid'],
    'model': ['threelayers'],
    'batch_size': [2048],
    'impute_nn': ['yes'],
    'keras_tuner': ['no'],
    'epochs': [epochs],
    'numbr_subjects': [num_subjects],
    'task': [2],
    'task2_activation': ['sigmoid'],
    'task2_loss': ['dice'],
    'with_time': ['yes'],
    'collapse_time': ['yes'],
}
search_space_dict_lin = {
    'nan_handling': ['iterative'],
    'standardizer': ['minmax'],
    'model': ['lin_reg'],
    'batch_size': [2048],
    'impute_nn': ['no', 'yes'],
    'keras_tuner': ['no'],
    'numbr_subjects': [num_subjects],
    'task3_loss': ['huber'],
    'task': [3],
    'with_time': ['no'],
    'collapse_time': ['yes'],
}

sample = pd.read_csv('sample.csv')
if not os.path.isfile('final.csv'):
    sample.to_csv('final.csv', index=False)

y_train_df = pd.read_csv('train_labels.csv').sort_values(by='pid')
X_train_df = pd.read_csv('train_features.csv').sort_values(by='pid')
X_final_df = pd.read_csv('test_features.csv')

def test_model(params, X_train_df, y_train_df, X_final_df, params_results_df):
    print('\n', params)
    train_path = 'nan_handling/train{}_{}.csv'.format(num_subjects, params['nan_handling'])
    test_path = 'nan_handling/test{}_{}.csv'.format(num_subjects, params['nan_handling'])
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        X_train_df = pd.read_csv(train_path)
        X_final_df = pd.read_csv(test_path)
    else:
        X_train_df = utils.handle_nans(X_train_df, params, seed)
        X_train_df.to_csv(train_path, index=False)
        X_final_df = utils.handle_nans(X_final_df, params, seed)
        X_final_df.to_csv(test_path, index=False)

    if params['collapse_time'] == 'yes':
        if params['with_time'] == 'no':
            x_train = X_train_df.groupby('pid').mean().reset_index()
            x_final = X_final_df.groupby('pid').mean().reset_index()
        if params['with_time'] == 'yes':
            x_final = X_final_df.groupby('pid').mean().reset_index()
            x_train = X_train_df.groupby('pid').mean().reset_index()
        input_shape = x_train.shape
    if params['collapse_time'] == 'no':
        x_train = []
        for i, subject in enumerate(list(dict.fromkeys(X_train_df['pid'].values.tolist()))):
            if X_train_df.loc[X_train_df['pid'] == subject].values[:, 1:].shape[0] > 12:
                raise Exception('more than 12 time-points')
            if subject in list(dict.fromkeys(y_train_df['pid'].values.tolist())):
                if params['with_time'] == 'yes':
                    x_train.append(X_train_df.loc[X_train_df['pid'] == subject].values[:, 1:])
                elif params['with_time'] == 'no':
                    x_train.append(X_train_df.loc[X_train_df['pid'] == subject].values[:, 2:])
            else:
                print(subject, 'not in y_train')
        input_shape = x_train[0].shape

    if params['collapse_time'] == 'yes':
        y_train1 = y_train_df.iloc[:,1:12]
        y_train2 = y_train_df.iloc[:, 11]
        y_train3 = y_train_df[:, 12:]
    else:
        y_train1 = list(y_train_df.values[:, 1:11])
        y_train2 = list(y_train_df.values[:, 11])
        y_train3 = list(y_train_df.values[:, 12:])

    if params['collapse_time'] == 'no':
        x_final = []
        for i, subject in enumerate(list(dict.fromkeys(X_final_df['pid'].values.tolist()))):
            if X_final_df.loc[X_final_df['pid'] == subject].values[:, 1:].shape[0] > 12:
                raise Exception('more than 12 time-points')
            if params['with_time'] == 'yes':
                x_final.append(X_final_df.loc[X_final_df['pid'] == subject].values[:, 1:])
            if params['with_time'] == 'no':
                x_final.append(X_final_df.loc[X_final_df['pid'] == subject].values[:, 2:])

    if params['task'] == 1:
        print('\n\n**** Training model1 **** \n\n')
        loss = params['task1_loss']
        if loss == 'dice':
            loss = dice_coef_loss
        model1, score1, scaler = utils.train_model(params, input_shape, x_train, y_train1, loss, epochs, seed, 1,
                                                   y_train_df)
        score2 = score3 = np.nan
    if params['task'] == 2:
        loss = params['task2_loss']
        if loss == 'dice':
            loss = dice_coef_loss
        print('\n\n**** Training model2 **** \n\n')
        model2, score2, scaler = utils.train_model(params, input_shape, x_train, y_train2, loss, epochs, seed, 2,
                                                   y_train_df)
        score1 = score3 = np.nan
    if params['task'] == 3:
        loss = params['task3_loss']
        if loss == 'dice':
            loss = dice_coef_loss
        elif loss == 'huber':
            loss = 'huber_loss'
        print('\n\n**** Training model3 **** \n\n')
        model3, score3, scaler = utils.train_model(params, input_shape, x_train, y_train3, loss, epochs, seed, 3,
                                                   y_train_df)
        score1 = score2 = np.nan

    if not params['standardizer'] == 'none':
        x_final, _ = utils.scaling(x_final, params, scaler)
    final_dataset = tf.data.Dataset.from_tensor_slices(x_final)
    final_dataset = final_dataset.batch(batch_size=params['batch_size'])

    final_df = pd.read_csv('final.csv')

    if params['task'] == 1 and (not params_results_df['task1'].values.tolist() or not np.max(
            params_results_df['task1'].values.tolist()) or score1 > np.max(params_results_df['task1'].values.tolist())):
        print('\n\n**** Writing to final for task1 **** \n\n')
        prediction1 = model1.predict(final_dataset)
        prediction_df = pd.DataFrame(prediction1, columns=y_train_df.columns[1:11])
        for col in prediction_df.columns:
            final_df[col] = prediction_df[col]
        final_df.reindex(columns=sample.columns)
        final_df.to_csv('final.csv', index=False)
    elif params['task'] == 1:
        print('\n\n**** Not writing to final for task1, current score: {} is smaller than {} **** \n\n'.format(score1,
                                                                                                               np.max(
                                                                                                                   params_results_df[
                                                                                                                       'task1'].values.tolist())))
    if params['task'] == 2 and (not params_results_df['task2'].values.tolist() or not np.max(
            params_results_df['task2'].values.tolist()) or score2 > np.max(params_results_df['task2'].values.tolist())):
        print('\n\n**** Writing to final for task2 **** \n\n')
        prediction2 = model2.predict(final_dataset)
        prediction_df = pd.DataFrame(prediction2, columns=[y_train_df.columns[11]])
        for col in prediction_df.columns:
            final_df[col] = prediction_df[col]
        final_df.reindex(columns=sample.columns)
        final_df.to_csv('final.csv', index=False)
    elif params['task'] == 2:
        print('\n\n**** Not writing to final for task2, current score: {} is smaller than {} **** \n\n'.format(score2,
                                                                                                               np.max(
                                                                                                                   params_results_df[
                                                                                                                       'task2'].values.tolist())))
    if params['task'] == 3 and (not params_results_df['task3'].values.tolist() or not np.max(
            params_results_df['task3'].values.tolist()) or score3 > np.max(params_results_df['task3'].values.tolist())):
        print('\n\n**** Writing to final for task3 **** \n\n')
        if params['model'].startswith('lin'):
            prediction3 = model3.predict(x_final)
        else:
            prediction3 = model3.predict(final_dataset)
        prediction_df = pd.DataFrame(prediction3, columns=y_train_df.columns[12:])
        for col in prediction_df.columns:
            final_df[col] = prediction_df[col]
        final_df.reindex(columns=sample.columns)
        final_df.to_csv('final.csv', index=False)
    elif params['task'] == 3:
        print('\n\n**** Not writing to final for task3, current score: {} is smaller than {} **** \n\n'.format(score3,
                                                                                                               np.max(
                                                                                                                   params_results_df[
                                                                                                                       'task3'].values.tolist())))

    return score1, score2, score3, np.mean([score1, score2, score3])


for search_space_dict in [search_space_dict_lin, search_space_dict_task1, search_space_dict_task2]:
    if not os.path.isfile('temp/params_results.csv'):
        columns = [key for key in search_space_dict.keys()]
        for key in ['task1', 'task2', 'task3', 'score']:
            columns.append(key)
        params_results_df = pd.DataFrame(columns=columns)
    else:
        params_results_df = pd.read_csv('temp/params_results.csv')
        for key in search_space_dict.keys():
            if not key in list(params_results_df.columns):
                params_results_df[key] = np.nan
    search_space = list(ParameterGrid(search_space_dict))
    search_space = random.sample(search_space, len(search_space))
    for params in tqdm(search_space):
        temp_df = params_results_df.loc[functools.reduce(operator.and_, (
            params_results_df['{}'.format(item)] == params['{}'.format(item)] for item in
            search_space_dict.keys())), 'task{}'.format(params['task'])]
        not_tested = temp_df.empty or temp_df.isna().all()
        if not_tested or test == True:
            if params['impute_nn'] == 'yes':
                if not os.path.isdir('imputed_data'):
                    os.mkdir('imputed_data')
                if not os.path.isfile('imputed_data/xtrain_imputedNN{}.csv'.format(num_subjects)):
                    X_train_df = utils.impute_NN(X_train_df)
                    X_train_df.to_csv('imputed_data/xtrain_imputedNN{}.csv'.format(num_subjects), index=False)
                    X_final_df = utils.impute_NN(X_final_df)
                    X_final_df.to_csv('imputed_data/xtest_imputedNN{}.csv'.format(num_subjects), index=False)
                else:
                    X_train_df = pd.read_csv('imputed_data/xtrain_imputedNN{}.csv'.format(num_subjects))
                    X_final_df = pd.read_csv('imputed_data/xtest_imputedNN{}.csv'.format(num_subjects))
            df = pd.DataFrame.from_records([params])
            scores = test_model(params, X_train_df, y_train_df, X_final_df, params_results_df)
            for column, score in zip(['task1', 'task2', 'task3', 'score'], scores):
                if type(score) == list:
                    df[column] = -1
                    df[column] = df[column].astype('object')
                    df.at[0, column] = score
                else:
                    df[column] = score
            print(df)
            params_results_df = params_results_df.append(df, sort=False)
        else:
            print('already tried this combination: ', params)
        if not test == True:
            params_results_df.to_csv('temp/params_results.csv', index=False)

# best = fmin(fn = test_model, space = search_space, algo = hp.tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)
# pd.DataFrame(best, index=[0]).to_csv('best_params.csv')
