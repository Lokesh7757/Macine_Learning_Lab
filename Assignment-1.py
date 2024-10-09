import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


os.chdir(r"F:\B_37")
data_csv=pd.read_csv('ASSIGNMENT 1.csv',na_values=["?"])
print(data_csv)
print("----------------------------------------------------------------")
data_csv.info()
print("----------------------------------------------------------------")
data_csv.isna().sum()
print(data_csv.isna().sum())
print("----------------------------------------------------------------")

print(data_csv.isnull().sum())
print("----------------------------------------------------------------")

missing=data_csv[data_csv.isnull().any(axis=1)]
print(missing)
print("----------------------------------------------------------------")
data_csv.describe()
print(data_csv.describe())
print("----------------------------------------------------------------")

data_csv['Age'].fillna(data_csv["Age"].mean(),inplace=True)
data_csv.isnull().sum()
print(data_csv)
print("----------------------------------------------------------------")

data_csv['Income'].fillna(data_csv['Income'].median(),inplace=True)
data_csv.isnull().sum()
print(data_csv)
print("----------------------------------------------------------------")

data_csv['Region'].fillna(data_csv['Region'].mode()[0],inplace=True)

data_csv.isnull().sum()
print(data_csv)

data_csv["Region"].value_counts().index[0]
data_csv["Region"].fillna(data_csv["Region"].value_counts().index[0])
print("-----------------data---------------------------")
print(data_csv)

df = pd.DataFrame(data_csv)


df.fillna({'Age': df['Age'].mean(), 'Income': df['Income'].mean()}, inplace=True)


label_encoder = LabelEncoder()
df['Online Shopper'] = label_encoder.fit_transform(df['Online Shopper'])


one_hot_encoder = OneHotEncoder()
region_encoded = one_hot_encoder.fit_transform(df[['Region']]).toarray()
region_encoded_df = pd.DataFrame(region_encoded, columns=one_hot_encoder.get_feature_names_out(['Region']))


df = pd.concat([df, region_encoded_df], axis=1).drop(['Region'], axis=1)


print("Encoded DataFrame:")
print(df)
print()

# Perform train-test split
X = df.drop("Online Shopper", axis=1)
y = df["Online Shopper"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print("Train-Test Split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

