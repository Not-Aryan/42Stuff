import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('/nfs/2020/a/ajain/PycharmProjects/MachineLearning/ML-3/dataset.csv')
# print(dataset.shape)
# dataset.plot(x='YearsExperience', y='Salary', style='o')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#Ask how to space out matplotlib

X = dataset['YearsExperience'].values.reshape(-1,1)
y = dataset['Salary'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(X_test)
print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

plt.scatter(X_test, y_test,  color='red')
plt.plot(X_test, y_pred, color='blue', linewidth=2)
plt.show()

# diabetes_y_pred = regressor.predict(X_test)
#
# plt.scatter(X_test, y_test,  color='red')
#
#
# plt.plot(X_test, y_test, color='blue', linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()






