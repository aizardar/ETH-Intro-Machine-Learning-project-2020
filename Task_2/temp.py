import pandas as pd
import tensorflow as tf
from pandas_profiling import ProfileReport
from matplotlib import pyplot as plt
import seaborn as sns



final_df  = pd.read_csv('predictions_.csv')
final_df.to_csv('prediction_.zip', index = False, float_format='%.3f', compression='zip')

# temp = pd.read_csv('xtrain_imputedNN-1.csv')
# prof = ProfileReport(temp, minimal=True)
#
#
#
#
# y_train_df = pd.read_csv('train_labels.csv')
# prof = ProfileReport(y_train_df)
# prof.to_file(output_file='train_labels.html')
#
#
# y_train_df = y_train_df.iloc[:, :10 + 1]
#
# x_train_df = pd.read_csv('train_features.csv')
# colormap = plt.cm.RdBu
# plt.figure(figsize=(32,32))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(x_train_df.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.savefig('temp/train_correlation')



# prof = ProfileReport(x_train_df, minimal= True)
# prof.to_file(output_file='pandas_profiling/train_minimal.html')



#
# new_df = x_train_df.dropna(axis = 0, how = 'any')
# prof = ProfileReport(new_df)
# prof.to_file(output_file='nonan.html')


