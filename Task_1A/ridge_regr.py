import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
print(train.head())
y = np.array(train['y'].tolist())
X = train.iloc[:, 2:].values

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(10,5))         # Sample figsize in inches
sns.heatmap(train.corr(), linewidths=.5, ax=ax)
plt.show()

df = pd.DataFrame(columns = ['alpha', 'RMSE', 'params', 'model'])

alphas = list(np.linspace(0, 100, 100000))
lasso_alphas = list(np.linspace(0, 100, 100000))
mean_RMSE = []
mean_RMSE_lasso = []
lasso_params = []
ridge_params = []

for i in range(len(alphas)):
    RMSE = []
    RMSE_lasso = []
    kf = KFold(n_splits = 10)
    kf.get_n_splits(X)
    for train_index, val_index in kf.split(X):
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        ridge_model = Ridge(alpha = alphas[i])
        ridge_model.fit(x_train, y_train)
        y_pred = ridge_model.predict(x_val)
        RMSE.append(mean_squared_error(y_val, y_pred)**0.5)
        lasso_model = Lasso(alpha = lasso_alphas[i])
        lasso_model.fit(x_train, y_train)
        y_pred = lasso_model.predict(x_val)
        RMSE_lasso.append(mean_squared_error(y_val, y_pred)**0.5)


    mean_RMSE.append(np.mean(RMSE))
    mean_RMSE_lasso.append((np.mean(RMSE_lasso)))
    lasso_params.append(lasso_model.coef_)
    df = df.append(pd.DataFrame([[alphas[i], np.mean(RMSE), ridge_model.coef_, 'ridge']], columns = ['alpha', 'RMSE', 'params', 'model']))
    df = df.append(pd.DataFrame([[lasso_alphas[i], np.mean(RMSE_lasso), lasso_model.coef_, 'lasso']], columns=['alpha', 'RMSE', 'params', 'model']))

df.to_csv('results.csv')