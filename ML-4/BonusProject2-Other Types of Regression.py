import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB




dataset = pd.read_csv('/nfs/2020/a/ajain/PycharmProjects/MachineLearning/ML-4/previous_users_dataset.csv')
dataset.head()

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


X = X.astype('float')
y = y.astype('float')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

# model = KNeighborsClassifier(n_neighbors=3)
model = GaussianNB()



model.fit(X_train, y_train)

X_set, y_set = X_test, y_test


X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, model.predict(
    np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=.75, cmap=ListedColormap(('red', 'green')))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=i)

plt.title('Support Vector Machine Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()