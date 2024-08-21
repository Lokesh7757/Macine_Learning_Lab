import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data for pizza prices
np.random.seed(42)
size = np.random.uniform(8, 20, 100)  # Pizza sizes between 8 and 20 inches
price = 5 + 2.5 * size + np.random.normal(0, 5, 100)  # Price based on size with some noise

# Create a DataFrame
data = pd.DataFrame({'Size': size, 'Price': price})

# Define features (X) and target (y)
X = data[['Size']]
y = data['Price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Fitted Line')
plt.xlabel('Pizza Size (in inches)')
plt.ylabel('Price')
plt.title('Pizza Price vs. Size')
plt.legend()
plt.show()
