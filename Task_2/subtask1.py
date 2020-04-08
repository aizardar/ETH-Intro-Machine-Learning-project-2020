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
from models import dice_coef_loss

hyperopt = False
if hyperopt:
    from hyperopt import Trials, fmin, STATUS_OK
    import hyperopt as hp
    bayes_trials = Trials()
    MAX_EVALS = 1000

"""
Predict whether medical tests are ordered by a clinician in the remainder of the hospital stay: 0 means that there will be no further tests of this kind ordered, 1 means that at least one of a test of that kind will be ordered. In the submission file, you are asked to submit predictions in the interval [0, 1], i.e., the predictions are not restricted to binary. 0.0 indicates you are certain this test will not be ordered, 1.0 indicates you are sure it will be ordered. The corresponding columns containing the binary groundtruth in train_labels.csv are: LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2.
10 labels for this subtask

Questions:
    - include time axis in training data?
    - use a SVM for each value to predict?
"""

seed = 100
batch_size = 64
num_subjects = -1         #number of subjects out of 18995
epochs = 50

search_space_dict = {
    'loss': ['dice', 'mean_squared_error', 'binary_crossentropy', 'categorical_hinge'],
    'nan_handling': ['minusone', 'zero', 'iterative'],
    'standardizer': ['none', 'RobustScaler', 'minmax', 'maxabsscaler', 'standardscaler'],
    'output_layer': ['sigmoid', 'linear'],
    'model': ['svm', 'threelayers'],
}

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

def test_model(params):
    print(params)
    if params['nan_handling'] == 'iterative':
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer, SimpleImputer
        except Exception as E:
            print(E)
            return np.nan

    loss = params['loss']
    if loss == 'dice':
        loss = dice_coef_loss
    X_train_df = pd.read_csv('train_features.csv').sort_values(by = 'pid')
    y_train_df = pd.read_csv('train_labels.csv').sort_values(by = 'pid')
    y_train_df = y_train_df.iloc[:num_subjects, :10 + 1]

    X_train_df = X_train_df.loc[X_train_df['pid'] < y_train_df['pid'].values[-1] + 1]
    X_train_df = utils.impute_NN(X_train_df)
    X_train_df.to_csv('temp/imputedNN.csv')
    X_train_df = utils.handle_nans(X_train_df, params, seed)

    """
    Scaling data
    """
    if not params['standardizer'] == 'none':
        scaler = utils.scaler(params)
        x_train_df = pd.DataFrame(data = scaler.fit_transform(X_train_df.values[:, 1:]), columns = X_train_df.columns[1:])
    else:
        x_train_df = X_train_df
    x_train_df.insert(0, 'pid', X_train_df['pid'].values)
    # x_train_df.to_csv('temp/taining_data.csv')

    x_train = []
    for i, subject in enumerate(list(dict.fromkeys(x_train_df['pid'].values.tolist()))):
        if X_train_df.loc[X_train_df['pid'] == subject].values[:, 1:].shape[0] > 12:
            raise Exception('more than 12 time-points')
        x_train.append(X_train_df.loc[X_train_df['pid'] == subject].values[:, 1:])
    input_shape = x_train[0].shape
    y_train = list(y_train_df.values[:, 1:])

    """
    Splitting the dataset into train 60%, val 30% and test 10% 
    """
    x_train, x_valtest, y_train, y_valtest = train_test_split(x_train, y_train, test_size=0.4, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(x_valtest, y_valtest, test_size=0.3, random_state=seed)

    """
    Making datasets
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size=batch_size).repeat()
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.shuffle(len(x_val)).batch(batch_size=batch_size).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(len(x_test)).batch(batch_size=batch_size)

    """
    Callbacks 
    """
    CB_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
        patience= 5,
        verbose=1,
        min_delta=0.0001,
        min_lr= 1e-6)

    CB_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        min_delta= 0.0001,
        patience= 10,
        mode='min',
        restore_best_weights=True)
    callbacks = [CB_es, CB_lr]

    if params['model'] == 'threelayers':
        model = threelayers(input_shape, loss, params['output_layer'])
    elif params['model'] == 'svm':
        model = models.svm(input_shape, loss, params['output_layer'])
    history = model.fit(train_dataset, validation_data = val_dataset, epochs = epochs, steps_per_epoch=len(x_train)//batch_size, validation_steps = len(x_train)//batch_size, callbacks = callbacks)
    print(model.summary())
    print('\nhistory dict:', history.history)

    prediction = model.predict(test_dataset)
    prediction_df = pd.DataFrame(prediction, columns= y_train_df.columns[1:])
    # prediction_df.to_csv('temp/result.csv')
    y_test_df = pd.DataFrame(y_test, columns= y_train_df.columns[1:])
    # y_test_df.to_csv('temp/ytrue.csv')
    roc_auc = np.mean([metrics.roc_auc_score(y_test_df[entry], prediction_df[entry]) for entry in y_train_df.columns[1:]])
    return roc_auc

for params in search_space:
    a = params_results_df.loc[(), 'roc_auc']
    temp_df = params_results_df.loc[functools.reduce(operator.and_, (params_results_df['{}'.format(item)] == params['{}'.format(item)] for item in search_space_dict.keys())), 'roc_auc']
    not_tested = temp_df.empty or temp_df.isna().all()
    if not_tested:
        df = pd.DataFrame.from_records([params])
        roc_auc = test_model(params)
        df['roc_auc'] = roc_auc
        params_results_df = params_results_df.append(df, sort= False)
    else:
        print('already tried this combination: ', params)
    df = pd.DataFrame.from_records([params])
    roc_auc = test_model(params)
    df['roc_auc'] = roc_auc
    params_results_df = params_results_df.append(df, sort= False)

    params_results_df.to_csv('temp/params_results.csv', index= False)




# best = fmin(fn = test_model, space = search_space, algo = hp.tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)
# pd.DataFrame(best, index=[0]).to_csv('best_params.csv')

