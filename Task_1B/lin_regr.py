import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from pandas_profiling import ProfileReport


normalize = True


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

# X_train_df = pd.DataFrame(X_train)
# prof = ProfileReport(X_train_df)
# prof.to_file(output_file='output.html')

ridge_alphas_range = (0, 100)
lasso_alphas_range = (0, 100)

for ind in range(5):
    models = [LinearRegression(normalize= normalize)]
    # alphas = list(np.linspace(ridge_alphas_range[0], ridge_alphas_range[1], 100))
    lasso_alphas = list(np.linspace(lasso_alphas_range[0], lasso_alphas_range[1], 1000))
    print(lasso_alphas)
    for i in range(len(lasso_alphas)):
        # results.append(Ridge(alpha = alphas[i], fit_intercept=False, normalize = normalize, max_iter= 1000000))
        # results.append(ElasticNet(alpha= alphas[i], l1_ratio= 0.5, fit_intercept=False, normalize= normalize))
        models.append(Lasso(alpha=lasso_alphas[i], fit_intercept=False, normalize= normalize, max_iter= 1000000))
    RMSES_val = []
    RMSES_train = []
    for i, model in enumerate(models):
        temp = cross_validate(X_train, y_train, model)
        RMSES_val.append(temp[0])
        RMSES_train.append(temp[1])
    best_idx = RMSES_val.index(min(RMSES_val))
    # print(lasso_alphas[best_idx])
    # ridge_alphas_range = (alphas[best_idx] - 10/(10**i), alphas[best_idx] + 10/(10**i))
    lasso_alphas_range = (lasso_alphas[best_idx] - 10 / (10 ** ind), lasso_alphas[best_idx] + 10 / (10 ** ind))
    print(ind, 20 / (10 ** ind))
    print(lasso_alphas_range)

model = models[best_idx]
print(model, RMSES_val[best_idx], RMSES_train[best_idx])
linear = model.fit(X_train, y_train)
pd.DataFrame(linear.coef_, dtype=float).to_csv('solution.csv', header = False, index=False)
