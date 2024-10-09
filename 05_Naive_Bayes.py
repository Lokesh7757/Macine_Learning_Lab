import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
from sklearn.naive_bayes import GaussianNB

print("Dataset: \n")
dataset = pd.read_csv("Tennis.csv")

x = dataset[['outlook','temp','humidity','windy']]
y = dataset['play']

x = pd.get_dummies(x)

print(dataset)

xtr,xt,ytr,yt = train_test_split(x,y,test_size=0.2,random_state=20)
print("\nTraining Dataset Independant Variable:\n")
print(xtr)
print("\nTraining Dataset Dependant Variable:\n")

print(ytr)
print("\nTesting Dataset Independant Variable:\n")
print(xt)
print("\nTesting Dataset Dependant Variable:\n")
print(yt)

nbc = GaussianNB()
nbc.fit(xtr,ytr)

y_pred = nbc.predict(xt)

print("\n\n\t\t\tClassification Report")
print(classification_report(yt,y_pred))
