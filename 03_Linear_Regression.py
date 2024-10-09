import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

dataset = pd.read_csv("Employee_Data.csv")
print(dataset)

x = dataset[['Age']]
y = dataset[['Salary']]

xtr,xt,ytr,yt = train_test_split(x,y,test_size=0.2,random_state=20)
print("\nTraining Dataset Independant Variable:\n")
print(xtr)
print("\nTraining Dataset Dependant Variable:\n")
print(ytr)
print("\nTesting Dataset Independant Variable:\n")
print(xt)
print("\nTesting Dataset Dependant Variable:\n")

print(yt)

model = LinearRegression()
model.fit(xtr,ytr)

print("\nPredicting Values from Testing Dataset :\n")
prediction = model.predict(xt)
print(prediction)

mse = mean_squared_error(yt,prediction)
print("\nMean Root Squared Error: ",np.sqrt(mse))

mae = mean_absolute_error(yt,prediction)
print("\nMean Absolute Error: ",mae)

sb.regplot(x=dataset[['Age']],y=dataset[['Salary']],ci=None)
plt.show()

plt.xlabel("Age")
plt.ylabel("Salary")
plt.scatter(xt,yt,color='red')
plt.plot(xt,prediction,color='blue')
plt.show()
