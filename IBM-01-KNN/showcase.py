import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

dataset = pd.read_csv('/nfs/2020/a/ajain/PycharmProjects/MachineLearning/IBM-01-KNN/iris.data')
print(dataset)

X = dataset.iloc[:, [2]].values
y = dataset.iloc[:, [3]].values

X = X.astype('float')
y = y.astype('float')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# encoder = preprocessing.LabelEncoder()
# X_train = encoder.fit_transform(X_train).reshape(-1,1)
# y_train = encoder.fit_transform(y_train).reshape(-1,1)
#
# y_test = encoder.fit_transform(y_test).reshape(-1,1)
# X_test = encoder.fit_transform(X_test).reshape(-1,1)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.scatter(X, y,  color='blue')
plt.show()



