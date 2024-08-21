import pandas as pd
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

dataset=load_digits()

#datset keys
print("Keys in datasets\n",dataset.keys())
print("\n")

#dataset data
print("Dataset data:\n",dataset.data)
print("\n")

#1st element of dataset
print("First element of data\n",dataset.data[0])
print("\n")

#to convert 1d flat array of element 1 into 2d array
print("Reshape Array\n",dataset.data[0].reshape(8,8))

#to plot array
plt.gray()
plt.matshow(dataset.data[0].reshape(8,8))
plt.show()


#target key
print(dataset.target[0])
print("\n")

#before applying pca
print("Number of pixels and column",dataset.data.shape)

#convert dataset into dataframe and use featurename as column name
print("DataFrame\n")
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
print(df.head())


#data info
print("\nData Describe")
print(df.describe())

X=df
y=dataset.target

#feature Scaling
scaler=StandardScaler()
ScaleX=scaler.fit_transform(X)
print("\nScaling X\n",ScaleX)

#splitting data
X_train, X_test, y_train, y_test=train_test_split(ScaleX,y,test_size=0.2,random_state=30)

#for classfication used logistic regression
model=LogisticRegression()
model.fit(X_train,y_train)#model traning
print("\nPrediction:",model.score(X_test,y_test))

#PCA
pca=PCA(0.95)#to get 0.95 features
X_pca=pca.fit_transform(X)
print("\nShape",X_pca.shape)
#final features
print("\n final features:",pca.n_components_)


X_train_pca, X_test_pca,y_train,y_test = train_test_split(X_pca, y, test_size=0.2,random_state=30)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)
print("PCA Train - Test")
print("\n")
print(model.score(X_test_pca,y_test))
print()

print("PCA - 5 components")
print("\n")
pca=PCA(n_components =5)
X_pca = pca.fit_transform(X)
print(X_pca.shape)
