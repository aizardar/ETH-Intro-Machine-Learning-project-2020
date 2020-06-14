import uuid
import os
import numpy as np
import pandas as pd
from preprocess import preprocess
from score_submission import get_score, get_all_scores
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid, GridSearchCV
from tqdm import tqdm
from utils import ExperimentLogger, mkdir, get_model

# %%
# Define parameter search spaces
# todo try SVM
task1_parameters_searchspace = {
    'model': ['random_forest_cl', 'grad_boosting'],
    'drop_features': [False, True],
    'with_grid_search_cv': [False,True],
}
task2_parameters_searchspace = {
    'model': ['random_forest_cl', 'grad_boosting'],
    'drop_features': [False, True],
    'with_grid_search_cv': [False,True],
}
task3_parameters_searchspace = {
    'model': ['lin_reg', 'random_forest_reg'],
    'drop_features': [False, True],
    'with_grid_search_cv': [False, True],
}

# In[ ]:
save_path = 'new_try_results/'
mkdir(save_path)
with_val_set = True

for task in [1, 2, 3]:
    # initialise experiment logger for each task
    if task == 1:
        search_space = list(ParameterGrid(task1_parameters_searchspace))

    elif task == 2:
        search_space = list(ParameterGrid(task2_parameters_searchspace))

    elif task == 3:
        search_space = list(ParameterGrid(task3_parameters_searchspace))

    for parameters in search_space:
        already_tried = False
        # initialise experiment logger for each task
        if task == 1:
            task1_parameters = parameters
            experiment_logger1 = ExperimentLogger(task1_parameters, task=1, save_path=save_path)
            if experiment_logger1.already_tried:
                print(f'already tried these parameters for task {task}: {parameters}')
                already_tried = True
            print('**** task1_parameters ', task1_parameters, '****')

        if task == 2:
            task2_parameters = parameters
            experiment_logger2 = ExperimentLogger(task2_parameters, task=2, save_path=save_path)
            if experiment_logger2.already_tried:
                print(f'already tried these parameters for task {task}: {parameters}')
                already_tried = True
            search_space = task2_parameters_searchspace
            print('**** task2_parameters ', task2_parameters, '****')

        if task == 3:
            task3_parameters = parameters
            experiment_logger3 = ExperimentLogger(task3_parameters, task=3, save_path=save_path)
            if experiment_logger3.already_tried:
                print(f'already tried these parameters for task {task}: {parameters}')
                already_tried = True
            search_space = task3_parameters_searchspace
            print('**** task3_parameters ', task3_parameters, '****')

        if not already_tried:
            train_features = pd.read_csv("train_features.csv").sort_values(by='pid')
            test = pd.read_csv("test_features.csv")
            train_labels = pd.read_csv("train_labels.csv").sort_values(by='pid')

            if with_val_set:
                train_features_all = train_features
                train_labels, val_labels = train_test_split(train_labels, test_size=0.2, random_state=42)
                train_features = train_features_all.merge(train_labels, on='pid')[
                    train_features_all.columns].sort_values(
                    by='pid')
                val_features = train_features_all.merge(val_labels, on='pid')[train_features_all.columns].sort_values(
                    by='pid')
                train_labels = train_labels.sort_values(by='pid')
                val_labels = val_labels.sort_values(by='pid')

            # In[ ]:
            if parameters['drop_features']:
                # Let's get rid of Hgb, HCO3, ABPd, and Bilirubin_direct from the training, validation and test data !

                train_features.drop('Bilirubin_direct', axis=1, inplace=True)
                train_features.drop('HCO3', axis=1, inplace=True)
                train_features.drop('Hgb', axis=1, inplace=True)
                train_features.drop('ABPd', axis=1, inplace=True)

                val_features.drop('Bilirubin_direct', axis=1, inplace=True)
                val_features.drop('HCO3', axis=1, inplace=True)
                val_features.drop('Hgb', axis=1, inplace=True)
                val_features.drop('ABPd', axis=1, inplace=True)

                test.drop('Bilirubin_direct', axis=1, inplace=True)
                test.drop('HCO3', axis=1, inplace=True)
                test.drop('Hgb', axis=1, inplace=True)
                test.drop('ABPd', axis=1, inplace=True)

            # In[ ]:

            val_labels, train_labels, new_test_features_imp, new_val_features_imp, new_train_features_imp = preprocess(
                parameters, with_val_set, train_features, val_features, test, val_labels, train_labels)

            feature_cols = new_train_features_imp.columns.values[
                (new_train_features_imp.columns.values != 'pid') & (new_train_features_imp.columns.values != 'Time')]

            X_train = new_train_features_imp[feature_cols]
            if with_val_set:
                X_val = new_val_features_imp[feature_cols]
            X_test = new_test_features_imp[feature_cols]
            # In[ ]:
            # Task 1 and Task 2

            TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
                     'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                     'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis']
            if not os.path.exists('predictions.csv'):
                # create submission dataframes that will be filled with the predictions
                df_submission = pd.DataFrame(new_test_features_imp.pid.values, columns=['pid'])
                new_df_submission = True
            else:
                df_submission = pd.read_csv('predictions.csv')
                new_df_submission = False

            if with_val_set:
                if new_df_submission or not os.path.exists('val_predictions.csv'):
                    df_submission_val = pd.DataFrame(new_val_features_imp.pid.values, columns=['pid'])
                else:
                    df_submission_val = pd.read_csv('val_predictions.csv')

            print("\n Training - Task 1 and 2 \n ")
            val_auc_scores = []
            for label in tqdm(TESTS):
                if label == 'LABEL_Sepsis' and task == 2:
                    # train, evaluate and predict for task 2
                    y_train_temp = train_labels[label]
                    if task2_parameters['with_grid_search_cv']:
                        model, param_grid = get_model(task2_parameters)
                        skf = StratifiedKFold(n_splits=5)
                        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, scoring='roc_auc',
                                             n_jobs=-1,
                                             verbose=1)
                    else:
                        model, _ = get_model(task2_parameters)
                    model.fit(X_train, y_train_temp)
                    """
                    predicting validation set
                    """
                    predictions_prob_val = model.predict_proba(X_val)
                    val_auc_score = roc_auc_score(val_labels[label], predictions_prob_val[:, 1])
                    experiment_logger2.df['{}_score'.format(label)] = val_auc_score
                    val_auc_scores.append(val_auc_score)
                    """
                    predicting on test set
                    """
                    # if previous results empty or if current score better than best previous score write to submission and to validation_pred
                    if experiment_logger2.previous_results_empty or val_auc_score > experiment_logger2.previous_results[
                        '{}_score'.format(label)].max():
                        predictions_prob_test = model.predict_proba(X_test)
                        predict_prob_val = pd.DataFrame(np.ravel(predictions_prob_val[:, 1]), columns=[label])
                        predict_prob_test = pd.DataFrame(np.ravel(predictions_prob_test[:, 1]), columns=[label])
                        if label in df_submission_val.columns:
                            df_submission_val.drop(label, axis=1, inplace=True)
                        if label in df_submission.columns:
                            df_submission.drop(label, axis=1, inplace=True)

                        df_submission_val = pd.concat([df_submission_val, predict_prob_val], axis=1, sort=False)
                        df_submission = pd.concat([df_submission, predict_prob_test], axis=1, sort=False)

                if not label == 'LABEL_Sepsis' and task == 1:
                    # train, evaluate and predict for task 1
                    y_train_temp = train_labels[label]
                    if task1_parameters['with_grid_search_cv']:
                        model, param_grid = get_model(task1_parameters)
                        skf = StratifiedKFold(n_splits=5)
                        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, scoring='roc_auc',
                                             n_jobs=-1,
                                             verbose=1)
                    else:
                        model, _ = get_model(task1_parameters)
                    model.fit(X_train, y_train_temp)
                    """
                    predicting validation set
                    """
                    predictions_prob_val = model.predict_proba(X_val)
                    val_auc_score = roc_auc_score(val_labels[label], predictions_prob_val[:, 1])
                    experiment_logger1.df['{}_score'.format(label)] = val_auc_score
                    val_auc_scores.append(val_auc_score)
                    """
                    predicting on test set
                    """
                    # if previous results empty or if current score better than best previous score write to submission and to validation_pred
                    if experiment_logger1.previous_results_empty or val_auc_score > experiment_logger1.previous_results[
                        '{}_score'.format(label)].max():
                        predictions_prob_test = model.predict_proba(X_test)
                        predict_prob_val = pd.DataFrame(np.ravel(predictions_prob_val[:, 1]), columns=[label])
                        predict_prob_test = pd.DataFrame(np.ravel(predictions_prob_test[:, 1]), columns=[label])
                        if label in df_submission_val.columns:
                            df_submission_val.drop(label, axis=1, inplace=True)
                        if label in df_submission.columns:
                            df_submission.drop(label, axis=1, inplace=True)

                        df_submission_val = pd.concat([df_submission_val, predict_prob_val], axis=1, sort=False)
                        df_submission = pd.concat([df_submission, predict_prob_test], axis=1, sort=False)

            # In[ ]:
            print('\n\n\n\n\n\n\n\n\n **** ', np.mean(val_auc_scores), '**** \n\n\n\n\n\n\n\n\n')
            # Task 3

            VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

            feature_cols = X_train.columns.values[
                (X_train.columns.values != 'pid') & (X_train.columns.values != 'Time')]

            X_train = X_train[feature_cols]
            X_test = X_test[feature_cols]
            X_val = X_val[feature_cols]
            print("\n Training - Task 3 \n ")

            for label in tqdm(VITALS):
                if task == 3:
                    # train, evaluate and predict for task 3
                    if task3_parameters['with_grid_search_cv']:
                        model, param_grid = get_model(task3_parameters)
                        skf = StratifiedKFold(n_splits=5)
                        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, scoring='roc_auc',
                                             n_jobs=-1)
                    else:
                        model, _ = get_model(task3_parameters)
                    y_temp = train_labels[label]
                    model.fit(X_train, y_temp)  # fitting the data
                    """
                    predicting validation set
                    """
                    y_pred_val = pd.DataFrame(model.predict(X_val), columns=[label])
                    val_score = r2_score(y_pred_val, val_labels[label])
                    experiment_logger3.df['{}_score'.format(label)] = val_score
                    """
                    predicting test set
                    """
                    # if previous results empty or if current score better than best previous score write to submission and to validation_pred
                    if experiment_logger3.previous_results_empty or val_score > experiment_logger3.previous_results[
                        '{}_score'.format(label)].max():
                        if label in df_submission_val.columns:
                            df_submission_val.drop(label, axis=1, inplace=True)
                        if label in df_submission.columns:
                            df_submission.drop(label, axis=1, inplace=True)
                        df_submission_val = pd.concat([df_submission_val, y_pred_val], axis=1, sort=False)
                        df_submission_val = pd.concat([df_submission_val, y_pred_val], axis=1, sort=False)
                        y_pred = pd.DataFrame(model.predict(X_test), columns=[label])
                        df_submission = pd.concat([df_submission, y_pred], axis=1, sort=False)

            # In[ ]:
            if len(val_labels.columns) == len(df_submission.columns):
                get_all_scores(val_labels, df_submission_val)
            try:  # todo if len(val_labels.columns) == len(df_submission) get all scores and print them + todo make unique uids!
                score = get_score(val_labels, df_submission_val, task)
            except:
                score = None
            if task == 1:
                experiment_logger1.df['task1_score'] = score
                experiment_logger1.save()

            if task == 2:
                experiment_logger2.df['task2_score'] = score
                experiment_logger2.save()

            if task == 3:
                experiment_logger3.df['task3_score'] = score
                experiment_logger3.save()

            df_submission.to_csv(r'predictions.csv', index=False, float_format='%.3f')
            df_submission_val.to_csv(r'val_predictions.csv', index=False, float_format='%.3f')
            print("All done !")

# In[ ]:
