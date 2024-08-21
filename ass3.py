import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import pandas as pd

# Load the Boston housing dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Print the dataset
print(df.head())

# Select 'RM' (average number of rooms per dwelling) as the feature and 'PRICE' as the target
X = df[['RM']]
y = df['PRICE']

# Scatter plot of the data
plt.scatter(X, y)
plt.xlabel("Average Number of Rooms")
plt.ylabel("Price")
plt.title("Scatter plot of Price vs. Average Number of Rooms")
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23)
print("Training features:\n", X_train.head())

# Reshape data
X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)

# Train a Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Get the model parameters
c = lr.intercept_
m = lr.coef_
print("Intercept (c):", c)
print("Coefficient (m):", m)

# Predict on the training set
y_pred_train = lr.predict(X_train)
print("Predicted values:\n", y_pred_train)

# Visualize the results
plt.scatter(X_train, y_train, color='blue', label='Actual')
plt.plot(X_train, y_pred_train, color='red', label='Fitted Line')
plt.xlabel("Average Number of Rooms")
plt.ylabel("Price")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

# Predict and evaluate on the test set
y_pred_test = lr.predict(X_test)
print("Test set predictions:\n", y_pred_test)
