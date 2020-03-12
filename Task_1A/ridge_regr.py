import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from  sklearn.preprocessing import scale
import numpy as np

train = pd.read_csv('train.csv')
print(train.head())
y = np.array(train['y'].tolist())
X = train.iloc[:, 2:].values
X = scale(X)    #standardization

alphas = [0.01, 0.1, 1, 10, 100]

mean_RMSE = []

for alpha in alphas:
    RMSE = []
    kf = KFold(n_splits= 10)
    kf.get_n_splits(X)
    for train_index, val_index in kf.split(X):
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model = Ridge(alpha = alpha)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        RMSE.append(mean_squared_error(y_val, y_pred)**0.5)

    mean_RMSE.append(np.mean(RMSE))

print(mean_RMSE)

pd.DataFrame(mean_RMSE).to_csv('solution.csv', index=False, header=False)