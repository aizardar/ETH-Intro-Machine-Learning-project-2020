import pandas as pd
import tensorflow as tf
from pandas_profiling import ProfileReport

y_train_df = pd.read_csv('train_labels.csv')
prof = ProfileReport(y_train_df)
prof.to_file(output_file='train_labels.html')


x_train_df = pd.read_csv('train_features.csv')
prof = ProfileReport(x_train_df, minimal= True)
prof.to_file(output_file='output.html')

new_df = x_train_df.fillna(-1)
prof = ProfileReport(x_train_df)
prof.to_file(output_file='output.html')
