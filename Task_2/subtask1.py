import pandas as pd
import tensorflow as tf
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
seed = 10
num_subjects = -1         #number of subjects out of 18995
epochs = 1
TRIALS = 50

search_space_dict = {
    'loss': ['dice','binary_crossentropy'],
    'nan_handling': ['minusone'],
    'standardizer': ['none'],
    'output_layer': ['sigmoid'],
    'model': ['threelayers'],
    'batch_size': [2048],
    'impute_nn': ['yes', 'no'],
    'keras_tuner': ['False'],
    'epochs': [epochs],
}
# test = True
# search_space_dict = {
#     'loss': ['binary_crossentropy'],
#     'nan_handling': ['minusone'],
#     'standardizer': ['none'],
#     'output_layer': ['sigmoid'],
#     'model': ['threelayers'],
#     'keras_tuner': ['False'],
#     'batch_size': [2048],
#     'impute_nn': ['yes', 'no']
# }

if not os.path.isfile('temp/params_results.csv'):
    columns = [key for key in search_space_dict.keys()]
    columns.append('roc_auc')
    params_results_df = pd.DataFrame(columns=columns)
else:
    params_results_df = pd.read_csv('temp/params_results.csv')
    for key in search_space_dict.keys():
        if not key in list(params_results_df.columns):
            params_results_df[key] = np.nan

search_space = list(ParameterGrid(search_space_dict))
y_train_df = pd.read_csv('train_labels.csv').sort_values(by='pid')
# y_train_df = y_train_df.iloc[:num_subjects, :10 + 1]
y_test_df = pd.read_csv('sample.csv').sort_values(by='pid')
# y_test_df = y_test_df.iloc[:num_subjects, :10 + 1]

X_train_df = pd.read_csv('train_features.csv').sort_values(by='pid')
X_train_df = X_train_df.loc[X_train_df['pid'] < y_train_df['pid'].values[-1] + 1]
X_test_df = pd.read_csv('train_features.csv').sort_values(by='pid')
X_test_df = X_test_df.loc[X_test_df['pid'] < y_test_df['pid'].values[-1] + 1]


def test_model(params, X_train_df, y_train_df, X_test_df, y_test_df, params_results_df):
    print('\n', params)
    train_path = 'nan_handling/train{}_{}.csv'.format(num_subjects, params['nan_handling'])
    test_path = 'nan_handling/test{}_{}.csv'.format(num_subjects, params['nan_handling'])
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        X_train_df = pd.read_csv(train_path)
        X_test_df = pd.read_csv(test_path)
    else:
        if params['nan_handling'] == 'iterative':
            try:
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer, SimpleImputer
            except Exception as E:
                print(E)
                return np.nan
        X_train_df = utils.handle_nans(X_train_df, params, seed)
        X_train_df.to_csv(train_path, index = False)
        X_test_df = utils.handle_nans(X_test_df, params, seed)
        X_test_df.to_csv(test_path, index = False)
    loss = params['loss']
    if loss == 'dice':
        loss = dice_coef_loss

    """
    Scaling data
    """
    if not params['standardizer'] == 'none':
        scaler = utils.scaler(params)
        x_train_df = pd.DataFrame(data = scaler.fit_transform(X_train_df.values[:, 1:]), columns = X_train_df.columns[1:])
        x_train_df.insert(0, 'pid', X_train_df['pid'].values)
    else:
        x_train_df = X_train_df
        x_test_df = X_test_df
    # x_train_df.to_csv('temp/taining_data.csv')

    x_train = []
    for i, subject in enumerate(list(dict.fromkeys(x_train_df['pid'].values.tolist()))):
        if X_train_df.loc[X_train_df['pid'] == subject].values[:, 1:].shape[0] > 12:
            raise Exception('more than 12 time-points')
        x_train.append(X_train_df.loc[X_train_df['pid'] == subject].values[:, 1:])
    input_shape = x_train[0].shape
    y_train1 = list(y_train_df.values[:, 1:11])
    y_train2 = list(y_train_df.values[:, 11])
    y_train3 = list(y_train_df.values[:, 12:])
    x_test = []
    for i, subject in enumerate(list(dict.fromkeys(x_test_df['pid'].values.tolist()))):
        if X_test_df.loc[X_test_df['pid'] == subject].values[:, 1:].shape[0] > 12:
            raise Exception('more than 12 time-points')
        x_test.append(X_test_df.loc[X_test_df['pid'] == subject].values[:, 1:])

    model1 = utils.train_model(params, input_shape, x_train, y_train1, loss, epochs, seed, task = 1)
    model2 = utils.train_model(params, input_shape, x_train, y_train2, loss, epochs, seed, task = 2)
    model3 = utils.train_model(params, input_shape, x_train, y_train3, loss, epochs, seed, task = 3)

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.batch(batch_size=1)

    prediction1 = model1.predict(test_dataset)
    prediction2 = model2.predict(test_dataset)
    prediction3 = model3.predict(test_dataset)
    prediction_df = pd.DataFrame(prediction1, columns= y_train_df.columns[1:11])
    prediction_df = prediction_df.append(pd.DataFrame(prediction2, columns = y_train_df.columns[11]))
    prediction_df = prediction_df.append(pd.DataFrame(prediction3, columns = y_train_df.columns[12:]))

    # y_test_df = pd.DataFrame(y_test, columns= y_train_df.columns[1:])
    # y_test_df.to_csv('temp/ytrue.csv')
    dice_score = [1 - dice(y_test_df[entry], np.where(prediction_df[entry] > 0.5, 1, 0)) for entry in y_train_df.columns[1:]]
    mean_dice = np.mean(dice_score)
    roc_auc = [metrics.roc_auc_score(y_test_df[entry], prediction_df[entry]) for entry in y_train_df.columns[1:]]
    mean_roc_auc = np.mean(roc_auc)
    score = get_score(y_test_df, prediction_df)
    if score > np.max(params_results_df['score'].values.to_list()):
        prediction_df.to_csv('prediction.csv'.format(np.round(score, 3)))
    return roc_auc, mean_roc_auc, dice_score, mean_dice, score

for params in tqdm(search_space):
    a = params_results_df.loc[(), 'roc_auc']
    temp_df = params_results_df.loc[functools.reduce(operator.and_, (params_results_df['{}'.format(item)] == params['{}'.format(item)] for item in search_space_dict.keys())), 'roc_auc']
    not_tested = temp_df.empty or temp_df.isna().all()
    if not_tested or test == True:
        if params['impute_nn'] == 'yes':
            if not os.path.isdir('imputed_data'):
                os.mkdir('imputed_data')
            if not os.path.isfile('imputed_data/xtrain_imputedNN{}.csv'.format(num_subjects)):
                X_train_df = utils.impute_NN(X_train_df)
                X_train_df.to_csv('imputed_data/xtrain_imputedNN{}.csv'.format(num_subjects), index=False)
                X_test_df = utils.impute_NN(X_test_df)
                X_test_df.to_csv('imputed_data/xtest_imputedNN{}.csv'.format(num_subjects), index=False)
            else:
                X_train_df = pd.read_csv('imputed_data/xtrain_imputedNN{}.csv'.format(num_subjects))
                X_test_df = pd.read_csv('imputed_data/xtest_imputedNN{}.csv'.format(num_subjects))
        df = pd.DataFrame.from_records([params])
        scores = test_model(params, X_train_df, y_train_df, X_test_df, y_test_df, params_results_df)
        for column, score in zip(['roc_auc', 'mean_roc_auc', 'dice_score', 'mean_dice'],scores):
            if type(score) == list:
                df[column] = -1
                df[column] = df[column].astype('object')
                df.at[0, column] = score
            else:
                df[column] = score
        print(df)
        params_results_df = params_results_df.append(df, sort= False)
    else:
        print('already tried this combination: ', params)
    if not test == True:
        params_results_df.to_csv('temp/params_results.csv', index= False)




# best = fmin(fn = test_model, space = search_space, algo = hp.tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)
# pd.DataFrame(best, index=[0]).to_csv('best_params.csv')

