import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('/nfs/2020/a/ajain/PycharmProjects/MachineLearning/ML-4/previous_users_dataset.csv')

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(dataset['Age'].values.reshape(-1,1))
y = scaler.fit_transform(dataset['EstimatedSalary'].values.reshape(-1,1))


# scaler = preprocessing.StandardScaler()
# scaled_x = scaler.fit_transform(dataset['Age'])
# scaled_y = scaler.fit_transform(dataset['Estimated Salary'])

plt.title('Logistical Regression(Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.plot(X, y, color='blue', style='o')
plt.show()


# feature_cols = ['Gender', 'Age', 'EstimatedSalary', 'UserID']
# X = scaled_df[feature_cols]
# y = scaled_df['Purchased']
#
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



