import pandas as pd
import tensorflow as tf
import models
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import os
from .utils import handle_nans, scaling, train_model, impute_NN
import functools, operator
from .models import dice_coef_loss
from tqdm import tqdm
import random
from .score_submission import get_score
import uuid

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
epochs = 5
tuner_trials = 50

search_space_dict_task1 = {
    'nan_handling': ['minusone', 'iterative'],
    'standardizer': ['RobustScaler'],
    'model': ['threelayers'],
    'batch_size': [2048],
    'impute_nn': ['yes'],
    'epochs': [epochs],
    'numbr_subjects': [num_subjects],
    'task': [1],
    'task1_activation': ['sigmoid'],
    'task1_loss': ['dice'],
    'with_time': ['no'],
    'collapse_time': ['no'],
    'tuner_trials': [tuner_trials],
    'uid': [uuid.uuid4()],
}
search_space_dict_task12 = {
    'nan_handling': ['mean', 'minusone', 'iterative'],
    'standardizer': ['RobustScaler', 'minmax'],
    # 'model': ['simple_conv_model', 'dense_model', 'threelayers', 'recurrent_net'],
    'model': ['lstm'],
    'output_layer': ['sigmoid', 'softmax'],
    'batch_size': [2048],
    'impute_nn': ['yes'],
    'epochs': [epochs],
    'numbr_subjects': [num_subjects],
    'task': [12],
    'task12_activation': ['sigmoid'],
    'task1_loss': ['dice'],
    'with_time': ['yes', 'no'],
    'collapse_time': ['no'],
    'tuner_trials': [tuner_trials],
    'uid': [uuid.uuid4()],
}

search_space_dict_task2 = {
    # 'loss': ['dice','binary_crossentropy'],
    'nan_handling': ['minusone'],
    'standardizer': ['minmax'],
    'output_layer': ['sigmoid'],
    'model': ['svm'],
    'batch_size': [2048],
    'impute_nn': ['yes'],
    'epochs': [epochs],
    'numbr_subjects': [num_subjects],
    'task': [2],
    'task2_activation': ['sigmoid'],
    'task2_loss': ['dice'],
    'with_time': ['yes'],
    'collapse_time': ['no'],
    'tuner_trials': [tuner_trials],
    'uid': [uuid.uuid4()],
}

search_space_dict_task3 = {
    # 'nan_handling': ['iterative', 'minusone', 'mean'],
    'nan_handling': ['minusone'],
    'task3_activation': ['linear'],
    'standardizer': ['RobustScaler'],
    # 'model': ['simple_conv_model', 'dense_model', 'threelayers', 'recurrent_net'],
    'model': ['threelayers'],
    'batch_size': [2048],
    'impute_nn': ['yes'],
    'epochs': [epochs],
    'task3_loss': ['mse', 'huber'],
    'numbr_subjects': [num_subjects],
    'task': [3],
    'with_time': ['yes', 'no'],
    'collapse_time': ['no'],
    'tuner_trials': [tuner_trials],
    'uid': [uuid.uuid4()],
}

search_space_dict_lin = {
    'nan_handling': ['iterative', 'minusone', 'mean'],
    # 'nan_handling': ['minusone'],
    'task3_activation': [None],
    'standardizer': ['none'],
    # 'model': ['simple_conv_model', 'dense_model', 'threelayers', 'recurrent_net'],
    'model': ['lin_reg'],
    'batch_size': [None],
    'impute_nn': ['yes', 'no'],
    'epochs': [None],
    'task3_loss': [None],
    'numbr_subjects': [num_subjects],
    'task': [3],
    'with_time': ['yes', 'no'],
    'collapse_time': ['yes'],
    'tuner_trials': [None],
    'uid': [uuid.uuid4()],
}

sample = pd.read_csv('sample.csv')
if not os.path.isfile('final.csv'):
    sample.to_csv('final.csv', index=False)

y_train_df = pd.read_csv('train_labels.csv').sort_values(by='pid')
X_train_df = pd.read_csv('train_features.csv').sort_values(by='pid')
X_final_df = pd.read_csv('test_features.csv')


def test_model(params, X_train_df, y_train_df, X_final_df, params_results_df):
    print('\n', params)
    # splitting the data into train and test
    y_train_df, y_test_df = train_test_split(y_train_df, test_size=0.2, random_state=seed)
    X_train_df, X_test_df = [X_train_df.merge(y_train_df, on='pid')[X_train_df.columns],
                             X_train_df.merge(y_test_df, on='pid')[X_train_df.columns]]

    train_path = 'nan_handling/train{}_{}.csv'.format(num_subjects, params['nan_handling'])
    x_final_path = 'nan_handling/final{}_{}.csv'.format(num_subjects, params['nan_handling'])
    x_test_path = 'nan_handling/test{}_{}.csv'.format(num_subjects, params['nan_handling'])

    if os.path.isfile(train_path) and os.path.isfile(x_final_path) and os.path.isfile(x_test_path):
        X_train_df = pd.read_csv(train_path)
        X_final_df = pd.read_csv(x_final_path)
        X_test_df = pd.read_csv(x_test_path)
    else:
        X_train_df, imp = handle_nans(X_train_df, params, seed)
        X_train_df.to_csv(train_path, index=False)
        if not params['nan_handling'] in ['zero', 'minusone']:
            X_final_df = pd.DataFrame(data=imp.transform(X_final_df), columns=X_final_df.columns)
            X_test_df = pd.DataFrame(data=imp.transform(X_test_df), columns=X_test_df.columns)
        else:
            X_final_df, _ = handle_nans(X_final_df, params, seed)
            X_test_df, _ = handle_nans(X_test_df, params, seed)
        X_final_df.to_csv(x_final_path, index=False)
        X_test_df.to_csv(x_test_path, index=False)

    if params['collapse_time'] == 'yes':
        x_train = X_train_df.groupby('pid').mean().reset_index()
        x_final = X_final_df.groupby('pid').mean().reset_index()
        x_test = X_test_df.groupby('pid').mean().reset_index()
        input_shape = x_train.shape

    if params['collapse_time'] == 'no':
        x_train = []
        for i, subject in enumerate(list(dict.fromkeys(y_train_df['pid'].values.tolist()))):
            if params['with_time'] == 'yes':
                x_train.append(X_train_df.loc[X_train_df['pid'] == subject].values[:, 1:])
            elif params['with_time'] == 'no':
                x_train.append(X_train_df.loc[X_train_df['pid'] == subject].values[:, 2:])

        x_test = []
        for i, subject in enumerate(list(dict.fromkeys(y_test_df['pid'].values.tolist()))):
            if params['with_time'] == 'yes':
                x_test.append(X_test_df.loc[X_test_df['pid'] == subject].values[:, 1:])
            elif params['with_time'] == 'no':
                x_test.append(X_test_df.loc[X_test_df['pid'] == subject].values[:, 2:])

        x_final = []
        for i, subject in enumerate(list(dict.fromkeys(X_final_df['pid'].values.tolist()))):
            if params['with_time'] == 'yes':
                x_final.append(X_final_df.loc[X_final_df['pid'] == subject].values[:, 1:])
            if params['with_time'] == 'no':
                x_final.append(X_final_df.loc[X_final_df['pid'] == subject].values[:, 2:])

        input_shape = x_train[0].shape

    if not params['standardizer'] == 'none':
        x_train, scaler =scaling(x_train, params)
        x_test, _ =scaling(x_test, params, scaler)
        x_final, _ =scaling(x_final, params, scaler)
    else:
        x_scaler = None

    if params['collapse_time'] == 'yes':
        if params['task'] == 12:
            y_train1 = y_train_df.iloc[:, 1:12]
        else:
            y_train1 = y_train_df.iloc[:, 1:11]
        y_train2 = y_train_df.iloc[:, 11]
        y_train3 = y_train_df.iloc[:, 12:]
    else:
        if params['task'] == 12:
            y_train1 = list(y_train_df.values[:, 1:12])
        else:
            y_train1 = list(y_train_df.values[:, 1:11])
        y_train2 = list(y_train_df.values[:, 11])
        y_train3 = list(y_train_df.values[:, 12:])

    test_score1 = test_score2 = test_score3 = test_score12 = np.nan

    if params['task'] == 1 or params['task'] == 12:
        print('\n\n**** Training model1 **** \n\n')
        loss = params['task1_loss']
        if loss == 'dice':
            loss = dice_coef_loss
        model1 =train_model(params, input_shape, x_train, y_train1, loss, epochs, seed, params['task'],
                                   y_train_df)

    if params['task'] == 2:
        loss = params['task2_loss']
        if loss == 'dice':
            loss = dice_coef_loss
        print('\n\n**** Training model2 **** \n\n')
        model2 = train_model(params, input_shape, x_train, y_train2, loss, epochs, seed, 2,
                                   y_train_df)

    if params['task'] == 3:
        loss = params['task3_loss']
        if loss == 'dice':
            loss = dice_coef_loss
        elif loss == 'huber':
            loss = 'huber_loss'
        print('\n\n**** Training model3 **** \n\n')
        model3 = train_model(params, input_shape, x_train, y_train3, loss, epochs, seed, 3,
                                   y_train_df)

    if params['model'] == 'resnet' or params['model'] == 'simple_conv_model':
        x_final = np.expand_dims(x_final, -1)
        x_test = np.expand_dims(x_test, -1)

    if not params['model'].startswith('lin'):
        final_dataset = tf.data.Dataset.from_tensor_slices(x_final)
        final_dataset = final_dataset.batch(batch_size=params['batch_size'])

        test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
        test_dataset = test_dataset.batch(batch_size=params['batch_size'])

    final_df = pd.read_csv('final.csv')
    if not os.path.exists('test_pred.csv'):
        test_df = pd.DataFrame(columns=sample.columns)
        test_df['pid'] = y_test_df['pid'].values
        test_df.to_csv('test_pred.csv')
    else:
        test_df = pd.read_csv('test_pred.csv')

    if params['task'] == 12:
        test_prediction12 = model1.predict(test_dataset)
        test_prediction_df = pd.DataFrame(test_prediction12, columns=y_test_df.columns[1:12])
        for col in test_prediction_df.columns:
            test_df[col] = test_prediction_df[col]
        test_df.reindex(columns=sample.columns)
        test_score12 = get_score(y_test_df, test_df)[0][0]

    if params['task'] == 12 and (not params_results_df['task12'].values.tolist() or np.isnan(np.max(
            params_results_df['task12'].values.tolist()))) or test_score12 > np.max(
        params_results_df['task12'].values.tolist()):

        test_df.to_csv('test_pred.csv', index=False)

        print('\n\n**** Writing to final for task12: new score {} is better than old score {} **** \n\n'.format(
            test_score12, np.max(params_results_df['task12'].values.tolist())))
        prediction1 = model1.predict(final_dataset)
        prediction_df = pd.DataFrame(prediction1, columns=y_train_df.columns[1:12])
        for col in prediction_df.columns:
            final_df[col] = prediction_df[col]
        final_df.reindex(columns=sample.columns)
        final_df.to_csv('final.csv', index=False)

    elif params['task'] == 12:
        print('\n\n**** Not writing to final for task12, current score: {} is smaller than {} **** \n\n'.format(
            test_score12,
            np.max(
                params_results_df[
                    'task12'].values.tolist())))

    if params['task'] == 1:
        test_prediction1 = model1.predict(test_dataset)
        test_prediction_df = pd.DataFrame(test_prediction1, columns=y_test_df.columns[1:11])
        for col in test_prediction_df.columns:
            test_df[col] = test_prediction_df[col]
        test_df.reindex(columns=sample.columns)
        test_score1 = get_score(y_test_df, test_df)[0][0]

    if params['task'] == 1 and (not params_results_df['task1'].values.tolist() or np.isnan(np.max(
            params_results_df['task1'].values.tolist())) or test_score1 > np.max(
        params_results_df['task1'].values.tolist())):

        test_df.to_csv('test_pred.csv', index=False)

        print('\n\n**** Writing to final for task1: new score {} is better than old score {} **** \n\n'.format(
            test_score1, np.max(params_results_df['task1'].values.tolist())))
        prediction1 = model1.predict(final_dataset)
        prediction_df = pd.DataFrame(prediction1, columns=y_train_df.columns[1:11])
        for col in prediction_df.columns:
            final_df[col] = prediction_df[col]
        final_df.reindex(columns=sample.columns)
        final_df.to_csv('final.csv', index=False)

    elif params['task'] == 1:
        print('\n\n**** Not writing to final for task1, current score: {} is smaller than {} **** \n\n'.format(
            test_score1,
            np.max(
                params_results_df[
                    'task1'].values.tolist())))
    if params['task'] == 2:
        test_prediction2 = model2.predict(test_dataset)
        test_prediction_df = pd.DataFrame(test_prediction2, columns=[y_test_df.columns[11]])
        for col in test_prediction_df.columns:
            test_df[col] = test_prediction_df[col]
        test_df.reindex(columns=sample.columns)
        test_score2 = get_score(y_test_df, test_df)[0][1]

    if params['task'] == 2 and (not params_results_df['task2'].values.tolist() or np.isnan(np.max(
            params_results_df['task2'].values.tolist())) or test_score2 > np.max(
        params_results_df['task2'].values.tolist())):

        test_df.to_csv('test_pred.csv', index=False)

        print('\n\n**** Writing to final for task2: new score {} is better than old score {} **** \n\n'.format(
            test_score2, np.max(params_results_df['task2'].values.tolist())))
        prediction2 = model2.predict(final_dataset)
        prediction_df = pd.DataFrame(prediction2, columns=[y_train_df.columns[11]])
        for col in prediction_df.columns:
            final_df[col] = prediction_df[col]
        final_df.reindex(columns=sample.columns)
        final_df.to_csv('final.csv', index=False)
    elif params['task'] == 2:
        print('\n\n**** Not writing to final for task2, current score: {} is smaller than {} **** \n\n'.format(
            test_score2,
            np.max(
                params_results_df[
                    'task2'].values.tolist())))
    if params['task'] == 3:

        if params['model'].startswith('lin'):
            test_prediction3 = model3.predict(x_test)
        else:
            test_prediction3 = model3.predict(test_dataset)
        test_prediction_df = pd.DataFrame(test_prediction3, columns=y_test_df.columns[12:])
        print(test_prediction_df.head(10))
        for col in test_prediction_df.columns:
            test_df[col] = test_prediction_df[col]
        test_df.reindex(columns=sample.columns)
        test_score3 = get_score(y_test_df, test_df)[0][2]

    if params['task'] == 3 and (not params_results_df['task3'].values.tolist() or np.isnan(np.max(
            params_results_df['task3'].values.tolist())) or test_score3 > np.max(
        params_results_df['task3'].values.tolist())):

        test_df.to_csv('test_pred.csv', index=False)

        print('\n\n**** Writing to final for task3: new score {} is better than old score {} **** \n\n'.format(
            test_score3, np.max(params_results_df['task3'].values.tolist())))
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
        print('\n\n**** Not writing to final for task3, current score: {} is smaller than {} **** \n\n'.format(
            test_score3,
            np.max(
                params_results_df[
                    'task3'].values.tolist())))

    return test_score1, test_score2, test_score3, test_score12


for search_space_dict in [search_space_dict_task3]:

    if not os.path.isfile('temp/params_results.csv'):
        columns = [key for key in search_space_dict.keys()]
        for key in ['task1', 'task2', 'task3', 'task12', 'score']:
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
            search_space_dict.keys() if not item == 'uid')), 'test_score{}'.format(params['task'])]
        not_tested = temp_df.empty or temp_df.isna().all()

        if not_tested or test == True:

            if params['impute_nn'] == 'yes':
                if not os.path.isdir('imputed_data'):
                    os.mkdir('imputed_data')
                if not os.path.isfile('imputed_data/xtrain_imputedNN{}.csv'.format(num_subjects)):
                    X_train_df = impute_NN(X_train_df)
                    X_train_df.to_csv('imputed_data/xtrain_imputedNN{}.csv'.format(num_subjects), index=False)
                    X_final_df = impute_NN(X_final_df)
                    X_final_df.to_csv('imputed_data/xtest_imputedNN{}.csv'.format(num_subjects), index=False)
                else:
                    X_train_df = pd.read_csv('imputed_data/xtrain_imputedNN{}.csv'.format(num_subjects))
                    X_final_df = pd.read_csv('imputed_data/xtest_imputedNN{}.csv'.format(num_subjects))
            df = pd.DataFrame.from_records([params])
            scores = test_model(params, X_train_df, y_train_df, X_final_df, params_results_df)

            for column, score in zip(
                    ['test_score1', 'test_score2',
                     'test_score3', 'test_score12'], scores):
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
