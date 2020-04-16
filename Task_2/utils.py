import pandas as pd
from sklearn import preprocessing
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import kerastuner
from kerastuner import RandomSearch
import tensorflow as tf
import models
import sklearn.metrics as metrics
from scipy.spatial.distance import dice


def handle_nans(X_train_df, params, seed):
    print('handling nans with method {}'.format(params['nan_handling']))
    if params['nan_handling'] == 'iterative':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(max_iter=10, random_state=seed)
        imp.fit(X_train_df)
        X_train_df = pd.DataFrame(data = imp.transform(X_train_df), columns = X_train_df.columns)
        # X_train_df.to_csv('temp/imputed_taining_data.csv')
    elif params['nan_handling'] == 'minusone':
        X_train_df = X_train_df.fillna(-1)
    elif params['nan_handling'] == 'zero':
        X_train_df = X_train_df.fillna(0)
    else:
        imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy=params['nan_handling'])
        imp.fit(X_train_df)
        X_train_df = pd.DataFrame(data = imp.transform(X_train_df), columns = X_train_df.columns)
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
    return scaler

def impute_NN(df):
    imputed_df = pd.DataFrame(columns=df.columns)
    imp = sklearn.impute.KNNImputer(n_neighbors=1)
    for pid in tqdm(np.unique(df['pid'].values)):
        temp_df = df.loc[df['pid'] == pid]
        temp_df2 = temp_df.dropna(axis = 'columns', how = 'all')
        imp.fit(temp_df2)
        temp_df2 = pd.DataFrame(data = imp.transform(temp_df2), columns = temp_df2.columns)
        for key in temp_df.columns:
            if temp_df[key].isna().all():
                temp_df2[key] = np.nan
        imputed_df = imputed_df.append(temp_df2, sort = True)
    imputed_df.reindex(columns = df.columns)
    return imputed_df

def train_model(params, input_shape, x_train, y_train, loss, epochs, seed, task, y_train_df):
    """
    Splitting the dataset into train 60%, val 30% and test 10%
    """
    x_train, x_valtest, y_train, y_valtest = train_test_split(x_train, y_train, test_size=0.4, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(x_valtest, y_valtest, test_size=0.3, random_state=seed)

    """
    Scaling data
    """
    if not params['standardizer'] == 'none':
        scaler = get_scaler(params)
        scaler.fit(np.concatenate(x_train)[:,1:])
        temp_train = np.concatenate(x_train)
        temp_train[:,1:] = scaler.transform(np.concatenate(x_train)[:,1:])
        x_train = np.vsplit(temp_train, temp_train.shape[0]/12)
        temp_val = np.concatenate(x_val)
        temp_val[:,1:] = scaler.transform(np.concatenate(x_val)[:,1:])
        x_val = np.vsplit(temp_val, temp_val.shape[0]/12)
        temp_test = np.concatenate(x_test)
        temp_test[:,1:] = scaler.transform(np.concatenate(x_test)[:,1:])
        x_test = np.vsplit(temp_test, temp_test.shape[0]/12)
    else:
        scaler = None

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
        patience= 5,
        verbose=1,
        min_delta=0.0001,
        min_lr= 1e-6)

    CB_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        min_delta= 0.0001,
        verbose= 1,
        patience= 10,
        mode='min',
        restore_best_weights=True)
    callbacks = [CB_es, CB_lr]
    if params['keras_tuner'] == 'False':
        if params['model'] == 'threelayers':
            model = models.threelayers(input_shape, loss, params['output_layer'], task)
        elif params['model'] == 'svm':
            model = models.svm(input_shape, loss, params['output_layer'])
        model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, steps_per_epoch=len(x_train) // params['batch_size'],
                  validation_steps=len(x_train) // params['batch_size'], callbacks=callbacks)
    elif params['keras_tuner'] == 'True':
        print(input_shape)
        tuner = RandomSearch(models.build_model, objective= kerastuner.Objective("val_auc", direction="min"), max_trials=TRIALS,
                             project_name='subtask1_results')
        tuner.search_space_summary()
        tuner.search(train_dataset, validation_data = val_dataset, epochs = epochs, steps_per_epoch=len(x_train)//params['batch_size'], validation_steps = len(x_train)//params['batch_size'], callbacks = callbacks)
        tuner.results_summary()

        # Retrieve the best model and display its architecture
        model = tuner.get_best_models(num_models=1)[0]

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.batch(batch_size=1)

    prediction = model.predict(test_dataset)
    if task == 1:
        prediction_df = pd.DataFrame(prediction, columns=y_train_df.columns[1:11])
        y_test_df = pd.DataFrame(y_test, columns=y_train_df.columns[1:11])
        roc_auc = [metrics.roc_auc_score(y_test_df[entry], prediction_df[entry]) for entry in
                   y_train_df.columns[1:11]]
        score = np.mean(roc_auc)
    if task == 2:
        prediction_df = pd.DataFrame(prediction, columns=[y_train_df.columns[11]])
        y_test_df = pd.DataFrame(y_test, columns=[y_train_df.columns[11]])
        score = metrics.roc_auc_score(y_test_df['LABEL_Sepsis'], prediction_df['LABEL_Sepsis'])
    if task == 3:
        prediction_df = pd.DataFrame(prediction, columns=y_train_df.columns[12:])
        y_test_df = pd.DataFrame(y_test, columns=y_train_df.columns[12:])
        score = np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(y_test_df[entry], prediction_df[entry])) for entry in y_train_df.columns[12:]])

    return model, score, scaler