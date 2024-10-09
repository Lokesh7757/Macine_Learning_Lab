#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
#importing dataset
data = pd.read_csv('Online_Shoping_Data.csv')
print("\n------------Data Imported Successfully")

#handling missing data
data["Income"].fillna(data["Income"].mean(),inplace=True)
data["Age"].fillna(data["Age"].median(),inplace=True)
print("\n------------Handling Missing Data")
print(data)
print("\n")

#handling categorial data
categories = data.groupby("Online Shopper").groups
le = LabelEncoder()
data["Online Shopper"] = le.fit_transform(data["Online Shopper"])
print("\n------------Data Categorised According to Online Shopper(Yes/No)")
print(data)
print("\n")

#partitioning data into testing and training
x = data[["Income","Age"]]
y = data["Online Shopper"].values
xtr,xt,ytr,yt = train_test_split(x,y,test_size=0.2)
print("\n------------Data is divided into testing and training successfully")

#Training Data
print("\n------------Training Dataset")
print(xtr)
print(ytr)

#Testing Data
print("\n------------Testing Dataset")
print(xt)
print(yt)
print("\n------------Performing Feature Scaling on data")

sc = StandardScaler()
df = data.filter(['Age','Income'],axis=1)
df = sc.fit_transform(df)
print(df)
