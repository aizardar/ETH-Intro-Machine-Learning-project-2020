import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from pandas_profiling import ProfileReport

def cross_validate(X, y, model):
    RMSE_val = []
    RMSE_train = []
    kf = KFold(n_splits= 10)
    kf.get_n_splits(X)
    for train_index, val_index in kf.split(X):
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(x_train, y_train)
        y_pred_val = model.predict(x_val)
        y_pred_train = model.predict(x_train)
        RMSE_val.append(mean_squared_error(y_val, y_pred_val)**0.5)
        RMSE_train.append(mean_squared_error(y_train, y_pred_train) ** 0.5)
    return np.mean(RMSE_val), np.mean(RMSE_train)

train_df = pd.read_csv('train.csv')
print(train_df.head())
print(train_df.describe())

x_train = train_df.iloc[:,2:].values
y_train = train_df['y']

X_train = np.column_stack([x_train, np.square(x_train, dtype= float), np.exp(x_train, dtype= float), np.cos(x_train, dtype= float), np.ones_like(y_train, dtype= float)])

X_train_df = pd.DataFrame(X_train)
prof = ProfileReport(X_train_df)
prof.to_file(output_file='output.html')
models = [LinearRegression(normalize= False)]
alphas = list(np.linspace(0.01, 100, 1000))
for alpha in alphas:
    models.append(Ridge(alpha = alpha, fit_intercept=False))
    models.append(ElasticNet(alpha= alpha, l1_ratio= 0.5, fit_intercept=False))
    models.append(Lasso(alpha=alpha, fit_intercept=False))
RMSES_val = []
RMSES_train = []
for i, model in enumerate(models):
    print(i, 'of ', len(models))
    temp = cross_validate(X_train, y_train, model)
    RMSES_val.append(temp[0])
    RMSES_train.append(temp[1])
best_idx = RMSES_val.index(min(RMSES_val))
model = models[best_idx]
print(model, RMSES_val[best_idx], RMSES_train[best_idx])
linear = model.fit(X_train, y_train)
pd.DataFrame(linear.coef_).to_csv('solution.csv', header = False, index=False)
