import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Preparing the data

dataset = pd.read_csv('/nfs/2020/a/ajain/PycharmProjects/MachineLearning/ML-4/previous_users_dataset.csv')

X = dataset['Age'].values.reshape(-1, 1)
y = dataset['EstimatedSalary'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(-2,3))
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

X = X.astype('float')
y = y.astype('float')


plt.title('Logistical Regression(Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

#Now performing the logistic regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train.astype('int'), y_train.ravel().astype('int'))
#Feature scaling is done after training cause it only accepts int values and this makes them floats

# scaler = MinMaxScaler(feature_range=(-2,3))
# X_test = scaler.fit_transform(X_test)
# y_test = scaler.fit_transform(y_test)

#Predicting
y_pred = logisticRegr.predict(X_test.astype('int'))

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

score = logisticRegr.score(X_test.astype('int'), y_test.astype('int'))
print('Score', score)

parameters = logisticRegr.coef_

plt.scatter(X_test, y_test,  color='red')
# plt.plot(X_test, y_pred, color='blue', linewidth=2)
plt.show()





