
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

training_set = pd.read_csv('Upvote_Train.csv')
test_set = pd.read_csv('Upvote_Test.csv')
test_set['Upvotes'] = 0

sample_submission = test_set[['ID', 'Upvotes']]
dataset = pd.concat([training_set, test_set], axis = 0)

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:, 6]

#Step 3: Data preparation
tag_dummy = pd.get_dummies(X['Tag'], drop_first = True)
X = pd.concat([X, tag_dummy], axis = 1)
X.drop(['Tag'],axis = 1, inplace = True)

#Splitting the dataset 
X_train = X.iloc[:330045,:]
y_train = Y[:330045]
X_test = X.iloc[330045:, : ]
y_test = Y.iloc[330045:,]

#Data Scaling
from sklearn.preprocessing import MinMaxScaler
num_vars = ['Reputation', 'Answers', 'Views']
sc_X = MinMaxScaler()
sc_y = MinMaxScaler()
X_train[num_vars] = sc_X.fit_transform(X_train[num_vars])
X_test[num_vars] = sc_X.fit_transform(X_test[num_vars])
y_train = sc_y.fit_transform(y_train.values.reshape(-1,1))

X_train.drop(['ID', 'Username'], axis = 1, inplace = True)
X_test.drop(['ID', 'Username'], axis = 1, inplace = True)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_test = sc_y.inverse_transform(y_pred)
sample_submission['Upvotes'] = y_test
sample_submission.loc[sample_submission['Upvotes'] < 0, 'Upvotes'] = 0
sample_submission.to_csv('sample_submission.csv', index = False)