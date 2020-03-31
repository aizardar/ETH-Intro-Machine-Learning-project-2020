import pandas as pd
import tensorflow as tf
from pandas_profiling import ProfileReport

x_train_df = pd.read_csv('train_features.csv')
x_train_df = x_train_df.iloc[:100, :]
prof = ProfileReport(x_train_df)
prof.to_file(output_file='output.html')