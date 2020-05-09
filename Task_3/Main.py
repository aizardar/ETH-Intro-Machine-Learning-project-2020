import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


codes = ['V', 'K', 'D', 'T', 'N', 'L', 'A', 'Y', 'P', 'C',
         'R', 'W', 'S', 'E', 'I', 'G', 'Q', 'H', 'F', 'M']

def create_dict(codes):
    char_dict = {}
    for index, val in enumerate(codes):
        char_dict[val] = index+1

    return char_dict

char_dict = create_dict(codes)


# Let's convert each character to an integer

def integer_encoding(data):
    
    
    encode_list = []
    for row in data['Sequence'].values:
        row_encode = []
        for code in row:
            row_encode.append(char_dict.get(code, 0))
        encode_list.append(np.array(row_encode))
  
    return encode_list



train_encode = integer_encoding(df_train) 
test_encode = integer_encoding(df_test) 

# Let's now one hot encode our training and test data

from sklearn.preprocessing import OneHotEncoder


ohe = OneHotEncoder()
ohe.fit(train_encode)
X_train_enc = ohe.transform(train_encode)
X_test_enc = ohe.transform(test_encode)


y_train = df_train.pop("Active")


clf = SVC(kernel = 'rbf', random_state=2020, gamma = 0.1 , C = 100)
clf.fit(X_train_enc, y_train)
y_pred = pd.DataFrame(clf.predict(X_test_enc))
y_pred.to_csv("prediction", header = False, index = False)
