import pandas as pd
from sklearn import preprocessing
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, HuberRegressor
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import kerastuner
from kerastuner import RandomSearch
import tensorflow as tf
import models
import sklearn.metrics as metrics
from scipy.spatial.distance import dice
from fancyimpute import KNN


def remove_outliers(x_train, y_train):
    x_train_means = pd.DataFrame(columns=x_train.columns)
    for i, subject in enumerate(list(dict.fromkeys(x_train['pid'].values.tolist()))):
        x_train_means.append(
            pd.DataFrame((x_train.loc[x_train['pid'] == subject][value].mean for value in x_train.columns[2:])))


def handle_nans(X_train_df, params, seed):
    print('handling nans with method {}'.format(params['nan_handling']))
    if params['nan_handling'] == 'iterative':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(max_iter=10, random_state=seed)
        imp.fit(X_train_df)
        X_train_df = pd.DataFrame(data=imp.transform(X_train_df), columns=X_train_df.columns)
        # X_train_df.to_csv('temp/imputed_taining_data.csv')
    elif params['nan_handling'] == 'minusone':
        X_train_df = X_train_df.fillna(-1)
    elif params['nan_handling'] == 'zero':
        X_train_df = X_train_df.fillna(0)
    elif params['nan_handling'] == 'KNN':
        X_train_df = KNN(k=3).fit_transform(X_train_df)
    elif params['nan_handling'] == 'iterative':
        imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy=params['nan_handling'])
        imp.fit(X_train_df)
        X_train_df = pd.DataFrame(data=imp.transform(X_train_df), columns=X_train_df.columns)
    return X_train_df


def get_scaler(params):
    if params['standardizer'] == 'RobustScaler':
        scaler = preprocessing.RobustScaler()
    elif params['standardizer'] == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    elif params['standardizer'] == 'maxabsscaler':
        scaler = preprocessing.MaxAbsScaler()
    elif params['standardizer'] == 'standardscaler':
        scaler = preprocessing.StandardScaler()
    else:
        scaler = None
    return scaler


def impute_NN(df):
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


def scaling(x, params, x_scaler=None):
    if x_scaler == None:
        x_scaler = get_scaler(params)
        if params['collapse_time'] == 'no':
            x_scaler.fit(np.concatenate(x)[:, 1:])
        if params['collapse_time'] == 'yes':
            x_scaler.fit(x)
    if params['collapse_time'] == 'no':
        x_temp = np.concatenate(x)
        x_temp[:, 1:] = x_scaler.transform(np.concatenate(x)[:, 1:])
        x = np.vsplit(x_temp, x_temp.shape[0] / x[0].shape[0])
    if params['collapse_time'] == 'yes':
        x = x_scaler.transform(x)

    return x, x_scaler


def train_model(params, input_shape, x_train, y_train, loss, epochs, seed, task, y_train_df):
    # todo this is not needed?
    # feature_cols = x_train.columns.values[
    #     (x_train.columns.values != 'pid') & (x_train.columns.values != 'Time')]
    #
    # x_trainval = x_trainval[feature_cols]
    # x_test = x_test[feature_cols]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)

    if params['model'] == 'resnet' or params['model'] == 'simple_conv_model':
        x_train = np.expand_dims(x_train, -1)
        x_val = np.expand_dims(x_val, -1)
        input_shape = x_train.shape[1:]
    """
    Making datasets
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size=params['batch_size']).repeat()
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.shuffle(len(x_val)).batch(batch_size=params['batch_size']).repeat()

    """
    Callbacks 
    """
    CB_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 patience=5,
                                                 verbose=1,
                                                 min_delta=0.0001,
                                                 min_lr=1e-6)

    CB_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=0.0001,
                                             verbose=1,
                                             patience=10,
                                             mode='min',
                                             restore_best_weights=True)
    callbacks = [CB_lr, CB_es]

    if params['keras_tuner'] == 'no':
        if params['model'].startswith('lin'):
            if params['model'] == 'lin_reg':
                model = LinearRegression().fit(x_train, y_train)
            if params['model'] == 'lin_huber':
                model = HuberRegressor().fit(x_train, y_train)
        else:
            if params['model'] == 'threelayers':
                model = models.threelayers(input_shape, loss, params['task{}_activation'.format(params['task'])], task)
            elif params['model'] == 'svm':
                model = models.svm(input_shape, loss, params['output_layer'])
            elif params['model'] == 'resnet':
                model = models.toy_ResNet(input_shape, loss)
            elif params['model'] == 'simple_conv_model':
                model = models.simple_conv_model(input_shape, loss)

            model.fit(train_dataset, validation_data=val_dataset, epochs=epochs,
                      steps_per_epoch=len(x_train) // params['batch_size'],
                      validation_steps=len(x_val) // params['batch_size'], callbacks=callbacks)

    elif params['keras_tuner'] == 'yes':
        if params['model'] == 'simple_conv_model':
            hypermodel = models.simple_conv_model(input_shape, loss, params['task{}_activation'.format(params['task'])],
                                                  task)
        elif params['model'] == 'dense_model':
            hypermodel = models.dense_model(input_shape, loss, params['task{}_activation'.format(params['task'])], task)
        tuner = RandomSearch(hypermodel, objective=kerastuner.Objective("val_auc", direction="max"),
                             max_trials=params['tuner_trials'],
                             project_name='{}_task{}'.format(params['model'], params['task']))
        tuner.search_space_summary()
        tuner.search(train_dataset, validation_data=val_dataset, epochs=epochs,
                     steps_per_epoch=len(x_train) // params['batch_size'],
                     validation_steps=len(x_val) // params['batch_size'], callbacks=callbacks)
        tuner.results_summary()

        # Retrieve the best model and display its architecture
        model = tuner.get_best_models(num_models=1)[0]

    return model
