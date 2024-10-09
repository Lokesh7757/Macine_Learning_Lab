import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report

url=r'F:\Machine Learning\New Folder.csv'
print("Loading dataset...")
df = pd.read_csv(url)
print("Dataset loaded successfully.\n")
print("First few rows of the dataset:")
print(df.head())

print("\nIdentifying non-numeric columns...")
non_numeric_columns = df.select_dtypes(include=['object']).columns
print(f"Non-numeric columns identified: {list(non_numeric_columns)}")
df = df.drop(columns=non_numeric_columns)
print("Non-numeric columns dropped.\n")
print("Dataset after dropping non-numeric columns:")
print(df.head())

print("\nFiltering rows where 'Prediction' column has values 0 or 1...")
df = df[df['Prediction'].isin([0, 1])]
print("Filtering complete.\n")
print("Dataset after filtering:")
print(df.head())

print("\nSplitting the data into features and labels...")
X = df.drop(columns=['Prediction'])
y = df['Prediction']
print("Features (X):")
print(X.head())
print("Labels (y):")
print(y.head())

print("\nSplitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Training features (X_train):")
print(X_train[:5])
print("Training labels (y_train):")
print(y_train[:5])
print("Testing features (X_test):")
print(X_test[:5])
print("Testing labels (y_test):")
print(y_test[:5])

print("\nStandardizing the data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data standardized.")
print("Standardized training features (X_train):")
print(X_train[:5])
print("Standardized testing features (X_test):")
print(X_test[:5])

print("\nInitializing and training the K-Nearest Neighbors model...")
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
print("Model training complete.\n")

print("Making predictions on the test set...")
y_pred = knn.predict(X_test)
print("Predictions complete.")
print("Predicted labels (y_pred):")
print(y_pred[:5])

print("\nEvaluating the model...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print('Classification Report:')
print(report)
