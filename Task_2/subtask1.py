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
num_subjects = 500         #number of subjects out of 18995
epochs = 1000
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
test = True
search_space_dict = {
    'loss': ['binary_crossentropy'],
    'nan_handling': ['minusone'],
    'standardizer': ['RobustScaler'],
    'output_layer': ['sigmoid'],
    'model': ['threelayers'],
    'impute_nn': ['no'],
    'keras_tuner': ['False'],
    'batch_size': [2048],
}

if not os.path.isfile('temp/params_results.csv'):
    columns = [key for key in search_space_dict.keys()]
    columns.append('score')
    params_results_df = pd.DataFrame(columns=columns)
else:
    params_results_df = pd.read_csv('temp/params_results.csv')
    for key in search_space_dict.keys():
        if not key in list(params_results_df.columns):
            params_results_df[key] = np.nan

search_space = list(ParameterGrid(search_space_dict))
y_train_df = pd.read_csv('train_labels.csv').sort_values(by='pid')

X_train_df = pd.read_csv('train_features.csv').sort_values(by='pid')
X_train_df = X_train_df.loc[X_train_df['pid'] < y_train_df['pid'].values[-1] + 1]
final_df = pd.read_csv('test_features.csv').sort_values(by='pid')

def test_model(params, X_train_df, y_train_df, final_df, params_results_df):
    print('\n', params)
    train_path = 'nan_handling/train{}_{}.csv'.format(num_subjects, params['nan_handling'])
    test_path = 'nan_handling/test{}_{}.csv'.format(num_subjects, params['nan_handling'])
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        X_train_df = pd.read_csv(train_path)
        final_df = pd.read_csv(test_path)
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
        final_df = utils.handle_nans(final_df, params, seed)
        final_df.to_csv(test_path, index = False)
    loss = params['loss']
    if loss == 'dice':
        loss = dice_coef_loss

    x_train = []
    for i, subject in enumerate(list(dict.fromkeys(X_train_df['pid'].values.tolist()))):
        if X_train_df.loc[X_train_df['pid'] == subject].values[:, 1:].shape[0] > 12:
            raise Exception('more than 12 time-points')
        if subject in list(dict.fromkeys(y_train_df['pid'].values.tolist())):
            x_train.append(X_train_df.loc[X_train_df['pid'] == subject].values[:, 1:])
        else:
            print(subject, 'not in y_train')
    input_shape = x_train[0].shape
    y_train1 = list(y_train_df.values[:, 1:11])
    y_train2 = list(y_train_df.values[:, 11])
    y_train3 = list(y_train_df.values[:, 12:])
    x_final = []
    for i, subject in enumerate(list(dict.fromkeys(final_df['pid'].values.tolist()))):
        if final_df.loc[final_df['pid'] == subject].values[:, 1:].shape[0] > 12:
            raise Exception('more than 12 time-points')
        x_final.append(final_df.loc[final_df['pid'] == subject].values[:, 1:])

    print('\n\n**** Training model1 **** \n\n')
    model1, score1, scaler1 = utils.train_model(params, input_shape, x_train, y_train1, loss, epochs, seed,1, y_train_df)
    print('\n\n**** Training model2 **** \n\n')
    model2, score2, scaler2 = utils.train_model(params, input_shape, x_train, y_train2, loss, epochs, seed,2, y_train_df)
    print('\n\n**** Training model3 **** \n\n')
    model3, score3, scaler3 = utils.train_model(params, input_shape, x_train, y_train3, loss, epochs, seed,3, y_train_df)

    x_final = scaler3.transform(x_final)
    final_dataset = tf.data.Dataset.from_tensor_slices(x_final)
    final_dataset = final_dataset.batch(batch_size=1)
    sample = pd.read_csv('sample.csv', index_col='pid')
    if not os.path.isfile('final.csv'):
        prediction1 = model1.predict(final_dataset)
        prediction2 = model2.predict(final_dataset)
        prediction3 = model3.predict(final_dataset)
        prediction_df = pd.DataFrame(prediction1, columns=y_train_df.columns[1:11])
        prediction_df[y_train_df.columns[11]] = prediction2
        prediction_df['pid'] = final_df['pid']
        prediction_df = prediction_df.join(pd.DataFrame(prediction3, columns=y_train_df.columns[12:]))
        prediction_df.reindex(columns = sample.columns)
        prediction_df.to_csv('final.csv')
    else:
        final_df = pd.read_csv('final.csv', index_col='pid')
        if not params_results_df.empty:
            if score1 > np.max(params_results_df['task1'].values.tolist()):
                prediction1 = model1.predict(final_dataset)
                prediction_df = pd.DataFrame(prediction1, columns=y_train_df.columns[1:11])
                for col in prediction_df.columns:
                    final_df[col] = prediction_df[col]
                final_df.reindex(columns=sample.columns)
                final_df.to_csv('final.csv')
            if score2 > np.max(params_results_df['task2'].values.tolist()):
                prediction2 = model2.predict(final_dataset)
                prediction_df = pd.DataFrame(prediction2, columns=y_train_df.columns[11])
                for col in prediction_df.columns:
                    final_df[col] = prediction_df[col]
                final_df.reindex(columns=sample.columns)
                final_df.to_csv('final.csv', index = False)
            if score3 > np.max(params_results_df['task3'].values.tolist()):
                prediction3 = model3.predict(final_dataset)
                prediction_df = pd.DataFrame(prediction3, columns=y_train_df.columns[12:])
                for col in prediction_df.columns:
                    final_df[col] = prediction_df[col]
                final_df.reindex(columns=sample.columns)
                final_df.to_csv('final.csv', index = False)
        print('scores: ', score1, score2, score3)
    return score1, score2, score3, np.mean([score1, score2, score3])


for params in tqdm(search_space):
    temp_df = params_results_df.loc[functools.reduce(operator.and_, (params_results_df['{}'.format(item)] == params['{}'.format(item)] for item in search_space_dict.keys())), 'score']
    not_tested = temp_df.empty or temp_df.isna().all()
    if not_tested or test == True:
        if params['impute_nn'] == 'yes':
            if not os.path.isdir('imputed_data'):
                os.mkdir('imputed_data')
            if not os.path.isfile('imputed_data/xtrain_imputedNN{}.csv'.format(num_subjects)):
                X_train_df = utils.impute_NN(X_train_df)
                X_train_df.to_csv('imputed_data/xtrain_imputedNN{}.csv'.format(num_subjects), index=False)
                final_df = utils.impute_NN(final_df)
                final_df.to_csv('imputed_data/xtest_imputedNN{}.csv'.format(num_subjects), index=False)
            else:
                X_train_df = pd.read_csv('imputed_data/xtrain_imputedNN{}.csv'.format(num_subjects))
                final_df = pd.read_csv('imputed_data/xtest_imputedNN{}.csv'.format(num_subjects))
        df = pd.DataFrame.from_records([params])
        scores = test_model(params, X_train_df, y_train_df, final_df, params_results_df)
        for column, score in zip(['task1', 'task2', 'task3', 'score'],scores):
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

