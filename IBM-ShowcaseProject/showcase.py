import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = pd.read_csv('/nfs/2020/a/ajain/PycharmProjects/MachineLearning/IBM-ShowcaseProject/FuelConsumption.csv')
print(dataset[['ENGINESIZE', 'FUELCONSUMPTION_HWY']])

plt.xlabel('ENGINESIZE')
plt.ylabel('FUELCOMSUMPTION_HWY')

X = dataset['ENGINESIZE'].values.reshape(-1,1)
y = dataset['FUELCONSUMPTION_HWY'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
# print(X_test)
# print(model.intercept_)
# print(model.coef_)

y_pred = model.predict(X)
print("R2-Score:", r2_score(y, y_pred))



# df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
# print(df)

plt.scatter(X, y,  color='blue')
plt.plot(X, y_pred, color='red', linewidth=2)
plt.show()




