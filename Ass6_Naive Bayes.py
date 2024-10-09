import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import sklearn.metrics

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('F:\Machine Learning\New Folder\Tennis.csv ' header=None, names=column_names)

df = df.dropna()

X = df.iloc[:, :-1].values
Y = df['MEDV'].values

X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').values
Y = pd.Series(Y).apply(pd.to_numeric, errors='coerce').values

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)  
Y = imputer.fit_transform(Y.reshape(-1, 1)).ravel()  

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5)


knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

best_k = 0
best_rmse = float('inf')
for i in range(1, 50):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("RMSE for k = {}: {:.2f}".format(i, rmse))
    if rmse < best_rmse:
        best_rmse = rmse
        best_k = i

print("Best k = {} with RMSE = {:.2f}".format(best_k, best_rmse))

SSSprint("RMSE for KNN Algorithm: {}".format(best_rmse))

