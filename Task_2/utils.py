import os
import kerastuner
import models
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from fancyimpute import KNN
from kerastuner import RandomSearch, Hyperband
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, HuberRegressor, SGDRegressor, Ridge
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import uuid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, \
    ExtraTreesClassifier, ExtraTreesRegressor


class ExperimentLogger():
    def __init__(self, params, task, save_path):
        self.params = params
        self.df = pd.DataFrame([params])
        self.task = task
        self.uid = self.create_uid()
        # self.df['uid'] = uuid.uuid4().hex
        self.save_path = save_path
        self.previous_results = self.load_previous_results()
        if (not self.previous_results_empty) and self.uid in self.previous_results['uid'].values:
            self.already_tried = True
        else:
            self.df['uid'] = self.uid
            self.already_tried = False

    def create_uid(self):
        values = self.df.iloc[0].values.tolist()
        uid = ''
        for elem in values:
            uid += str(elem)
        return uid

    def save(self):
        if not os.path.exists(os.path.join(self.save_path, f'experiment_logs{self.task}.csv')):
            self.df.to_csv(os.path.join(self.save_path, f'experiment_logs{self.task}.csv'), index=False)
        else:
            experiment_logs = pd.read_csv(os.path.join(self.save_path, f'experiment_logs{self.task}.csv'))
            experiment_logs = pd.concat([experiment_logs, self.df])
            experiment_logs.to_csv(os.path.join(self.save_path, f'experiment_logs{self.task}.csv'), index=False)

    def load_previous_results(self):
        if os.path.exists(os.path.join(self.save_path, f'experiment_logs{self.task}.csv')):
            self.previous_results_empty = False
            return pd.read_csv(os.path.join(self.save_path, f'experiment_logs{self.task}.csv'))
        else:
            self.previous_results_empty = True
            return 'empty'

class ExperimentLogger12(ExperimentLogger):
    def init(self):
        self.previous_results1 = self.load_previous_results(1)
        self.previous_results2 = self.load_previous_results(2)
        self.previous_results12 = self.load_previous_results(12)
        super().__init__()
    def load_previous_results(self, task):
        if os.path.exists(os.path.join(self.save_path, f'experiment_logs{task}.csv')):
            self.previous_results_empty = False
            return pd.read_csv(os.path.join(self.save_path, f'experiment_logs{task}.csv'))
        else:
            self.previous_results_empty = True
            return 'empty'





def get_model(params):
    if params['model'] == 'random_forest_cl':
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': np.linspace(100, 800, 4, dtype=int),
            'max_features': ['auto', 'log2', None],
            'criterion': ['gini', 'entropy']}
    elif params['model'] == 'grad_boosting':
        model = GradientBoostingClassifier()
        param_grid = {
            'n_estimators': np.linspace(100, 800, 4, dtype=int),
            'max_features': ['auto', 'log2', None],
            'criterion': ['gini', 'entropy']}
    elif params['model'] == 'random_forest_reg':
        param_grid = {
            'n_estimators': np.linspace(100, 800, 4, dtype=int),
            'max_features': ['auto', 'log2', None],
            'criterion': ['mse', 'mae']}
        model = RandomForestRegressor()
    elif params['model'] == 'lin_reg':
        param_grid = {
            'normalize': [True, False],
            'fit_intercept': [True, False],
        }
        model = LinearRegression()
    elif params['model'] == 'SGD_reg':
        param_grid = {
            'alpha': np.logspace(-10, -2, 10),
        }
        model = SGDRegressor()
    elif params['model'] == 'huber_reg':
        param_grid = {
            'epsilon': np.linspace(1.0, 3, 5),
            'alpha': np.linspace(0.00001, 0.1, 5),
            'fit_intercept': [True, False],
        }
        model = HuberRegressor()
    elif params['model'] == 'ridge_reg':
        param_grid = {
            'alpha': np.logspace(-10, -2, 10),
        }
        model = Ridge()
    elif params['model'] == 'extra_tree_cl':
        param_grid = {
            'n_estimators': np.linspace(100, 800, 4, dtype=int),
            'max_features': ['auto', 'log2', None],
            'criterion': ['gini', 'entropy']}
        model = ExtraTreesClassifier()
    elif params['model'] == 'extra_tree_reg':
        param_grid = {
            'n_estimators': np.linspace(100, 800, 4, dtype=int),
            'max_features': ['auto', 'log2', None],
            'criterion': ['gini', 'entropy']}
        model = ExtraTreesRegressor()
    elif params['model'] == 'SVC_rbf':
        model = SVC(kernel='rbf', random_state=42, probability=True)
    elif params['model'] == 'SVC_sig':
        model = SVC(kernel='sigmoid', random_state=42, probability=True)
    elif params['model'] == 'SVR_lin':
        model = SVR(kernel='linear')
    elif params['model'] == 'SVR_sig':
        model = SVR(kernel='sigmoid')
    elif params['model'] == 'SVR_rbf':
        model = SVR(kernel='rbf')
    elif params['model'] == 'SVR_poly':
        model = SVR(kernel='poly')

    if not params['with_grid_search_cv']:
        param_grid = None
    return model, param_grid


def bigprint(content):
    print(
        '\n\n\n\n\n\n\n\n\n *********************************************\n {} \n *********************************************\n\n\n\n\n\n\n\n\n'.format(
            content))


def get_feature_selector(params):
    if params['feature_selection'] == 'l1_SVC':
        return LinearSVC(penalty="l1", dual=False, max_iter=2000)
    # this doesn't work
    # if params['feature_selection'] == 'sgd_reg':
    #     return LinearSVR(penalty="l1", dual=False, max_iter=2000)
    else:
        assert 1, 'wrong input'


def mkdir(path):
    if not os.path.exists(path):
        print('creating dir ', path)
        os.makedirs(path)


def remove_outliers(x_train, y_train):
    x_train_means = pd.DataFrame(columns=x_train.columns)
    for i, subject in enumerate(list(dict.fromkeys(x_train['pid'].values.tolist()))):
        x_train_means.append(
            pd.DataFrame((x_train.loc[x_train['pid'] == subject][value].mean for value in x_train.columns[2:])))


def handle_nans(X_train_df, params, seed):
    print('handling nans with method {}'.format(params['nan_handling']))
    if params['nan_handling'] == 'iterative':
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(max_iter=10, random_state=seed)
        imp.fit(X_train_df)
        X_train_df = pd.DataFrame(data=imp.transform(X_train_df), columns=X_train_df.columns)
        # X_train_df.to_csv('temp/imputed_taining_data.csv')
    elif params['nan_handling'] == 'minusone':
        X_train_df = X_train_df.fillna(-1)
        imp = 0
    elif params['nan_handling'] == 'zero':
        X_train_df = X_train_df.fillna(0)
        imp = 0
    elif params['nan_handling'] == 'KNN':
        X_train_df = KNN(k=3).fit_transform(X_train_df)
        imp = 0
    else:
        imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy=params['nan_handling'])
        imp.fit(X_train_df)
        X_train_df = pd.DataFrame(data=imp.transform(X_train_df), columns=X_train_df.columns)

    return X_train_df, imp


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
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)

    if params['model'] in ['resnet', 'recurrent_net', 'simple_conv_model']:
        x_train = np.expand_dims(x_train, -1)
        x_val = np.expand_dims(x_val, -1)
        input_shape = x_train.shape[1:]
    """
    Making datasets
    """
    if not params['model'].startswith('lin'):
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size=params['batch_size']).repeat()
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.shuffle(len(x_val)).batch(batch_size=params['batch_size']).repeat()

    """
    Callbacks 
    """
    CB_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc" * (params['task'] == 1 or params['task'] == 12 or params['task'] == 2) + "val_mse" * (
                params['task'] == 3),
        patience=3,
        verbose=1,
        mode="max" * (params['task'] == 1 or params['task'] == 12 or params['task'] == 2) + "min" * (
                params['task'] == 3),
        min_delta=0.0001,
        min_lr=1e-6)

    CB_es = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc" * (params['task'] == 1 or params['task'] == 12 or params['task'] == 2) + "val_mse" * (
                params['task'] == 3),
        min_delta=0.00001,
        verbose=1,
        patience=10,
        mode="max" * (params['task'] == 1 or params['task'] == 12 or params['task'] == 2) + "min" * (
                params['task'] == 3),
        restore_best_weights=True)
    callbacks = [CB_lr, CB_es]
    # callbacks = [CB_lr]

    if params['model'] in ['lin_reg', 'lin_huber', 'threelayers', 'svm', 'resnet', 'lstm']:
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
            elif params['model'] == 'lstm':
                model = models.lstm(input_shape, loss, params, params['task'])

            model.fit(train_dataset, validation_data=val_dataset, epochs=epochs,
                      steps_per_epoch=len(x_train) // params['batch_size'],
                      validation_steps=len(x_val) // params['batch_size'], callbacks=callbacks)

    else:
        if params['model'] == 'simple_conv_model':
            hypermodel = models.simple_conv_model(input_shape, loss, params['task{}_activation'.format(params['task'])],
                                                  task)
        elif params['model'] == 'dense_model':
            hypermodel = models.dense_model(input_shape, loss, params['task{}_activation'.format(params['task'])], task)
        elif params['model'] == 'recurrent_net':
            hypermodel = models.recurrent_net(input_shape, loss, params['task{}_activation'.format(params['task'])],
                                              task)

        if not os.path.exists('tuner_trials'):
            os.mkdir('tuner_trials')
        # tuner = RandomSearch(hypermodel, objective=kerastuner.Objective(
        #     "val_auc" * (params['task'] == 1 or params['task'] == 12 or params['task'] == 2) + "val_mse" * (
        #             params['task'] == 3),
        #     direction="max"),
        #                   max_trials=params['tuner_trials'],
        #                   project_name='keras_tuner/{}_{}_task{}_{}'.format(params['model'], input_shape,
        #                                                                     params['task'], params['uid']))
        tuner = Hyperband(hypermodel, objective=kerastuner.Objective(
            "val_auc" * (params['task'] == 1 or params['task'] == 12 or params['task'] == 2) + "val_mse" * (
                    params['task'] == 3),
            direction="max"),
                          project_name='keras_tuner/{}_{}_task{}_{}'.format(params['model'], input_shape,
                                                                            params['task'], params['uid']),
                          max_epochs=epochs, factor=epochs // params['tuner_trials'])
        tuner.search_space_summary()
        tuner.search(train_dataset, validation_data=val_dataset, epochs=epochs,
                     steps_per_epoch=len(x_train) // params['batch_size'],
                     validation_steps=len(x_val) // params['batch_size'], callbacks=callbacks)
        tuner.results_summary()

        # Retrieve the best model and display its architecture
        model = tuner.get_best_models(num_models=1)[0]

    return model
