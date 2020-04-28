import pandas as pd

from sklearn.linear_model import LinearRegression

y_train_df = pd.read_csv('train_labels.csv')
x_train_df = pd.read_csv('train_features.csv')

x_train = x_train_df.iloc[:,2:]

x_train = [x_train[count * 12: (count + 1) * 12,:] for count in len(y_train_df)]

time = np.array(range(1,13))

for idx in range(len(x_train)):
    reg = LinearRegression()

    reg.fit(time, x_train[idx])

    x_time_data.append
