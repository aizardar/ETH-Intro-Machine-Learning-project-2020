import pandas as pd
import tensorflow as tf
from models import simple_model, threelayers
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn import preprocessing, svm
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from hyperopt import Trials,fmin,STATUS_OK
import hyperopt as hp
import os



"""
Predict whether medical tests are ordered by a clinician in the remainder of the hospital stay: 0 means that there will be no further tests of this kind ordered, 1 means that at least one of a test of that kind will be ordered. In the submission file, you are asked to submit predictions in the interval [0, 1], i.e., the predictions are not restricted to binary. 0.0 indicates you are certain this test will not be ordered, 1.0 indicates you are sure it will be ordered. The corresponding columns containing the binary groundtruth in train_labels.csv are: LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2.
10 labels for this subtask
"""

seed = 100
batch_size = 64
num_subjects = -1          #number of subjects out of 18995
epochs = 1000

# search_space = {
#     'loss': hp.hp.choice('loss', ['mean_squared_error', 'binary_crossentropy', 'categorical_hinge']),
#     'imputing': hp.hp.choice('imputing', [True, False]),
# }
search_space = {
    'loss': ['mean_squared_error', 'binary_crossentropy', 'categorical_hinge'],
    'nan_handling': ['impute', 'minusone', 'zero', 'mean', 'median', 'most_frequent'],
}

if not os.path.isfile('temp/params_results.csv'):
    columns = [key for key in search_space.keys()]
    columns.append('roc_auc')
    params_results_df = pd.DataFrame(columns=columns)
else:
    params_results_df = pd.read_csv('temp/params_results.csv')

search_space = list(ParameterGrid(search_space))

bayes_trials = Trials()
MAX_EVALS = 1000

def test_model(params):
    print(params)
    loss = params['loss']
    X_train_df = pd.read_csv('train_features.csv').sort_values(by= 'pid')
    y_train_df = pd.read_csv('train_labels.csv').sort_values(by= 'pid')
    y_train_df = y_train_df.iloc[:num_subjects, :10 + 1]

    X_train_df = X_train_df.loc[X_train_df['pid'] < y_train_df['pid'].values[-1] + 1]

    """
    instead of imputing I could also set all the nans to -1
    """
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
    else:
        imp = SimpleImputer(missing_values=np.nan, strategy=params['nan_handling'])
        imp.fit(X_train_df)
        X_train_df = pd.DataFrame(data = imp.transform(X_train_df), columns = X_train_df.columns)
    scaler = preprocessing.MinMaxScaler()
    x_train_df = pd.DataFrame(data = scaler.fit_transform(X_train_df.values[:, 1:]), columns = X_train_df.columns[1:])
    x_train_df.insert(0, 'pid', X_train_df['pid'].values)
    # x_train_df.to_csv('temp/taining_data.csv')

    x_train = []
    for i, subject in enumerate(list(dict.fromkeys(x_train_df['pid'].values.tolist()))):
        x_train.append(np.concatenate(X_train_df.loc[X_train_df['pid'] == subject].values[:, 1:]))
    input_shape = x_train[0].shape

    """
    Splitting the dataset into train 60%, val 30% and test 10% 
    """
    x_train, x_valtest, y_train, y_valtest = train_test_split(x_train, y_train_df.values[:, 1:], test_size=0.4, random_state=seed)
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

    model = threelayers(input_shape, loss)
    history = model.fit(train_dataset, validation_data = val_dataset, epochs = epochs, steps_per_epoch=input_shape[0]//batch_size, validation_steps = input_shape[0]//batch_size
                        , callbacks = callbacks
                        )
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
    try:
        temp = params_results_df.loc[(params_results_df['loss'] == params['loss']) & (params_results_df['nan_handling'] == params['nan_handling'])]
    except:
        df = pd.DataFrame.from_records([params])
        roc_auc = test_model(params)
        df['roc_auc'] = roc_auc
        params_results_df = params_results_df.append(df, sort= False)

        params_results_df.to_csv('temp/params_results.csv', index= False)




# best = fmin(fn = test_model, space = search_space, algo = hp.tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)
# pd.DataFrame(best, index=[0]).to_csv('best_params.csv')

