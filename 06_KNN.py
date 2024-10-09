import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Define the file path
file_path = r'F:\Machine Learning\New Folder.csv'  # Adjust this path if necessary

# Check if the file exists
if os.path.exists(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Explore the dataset
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Encode the label if necessary
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    # Ensure 'label' column is the target and the rest are features
    X = df.drop('label', axis=1)
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Best estimator
    knn = grid_search.best_estimator_

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate accuracy and precision
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)  # Assuming 'spam' is encoded as 1

    print(f'\nAccuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    # Classify a new email
    new_email_features = [[10, 1, 0]]  # Example feature values (modify as necessary)
    new_email_features = scaler.transform(new_email_features)
    new_email_prediction = knn.predict(new_email_features)

    print('\nNew Email Classification:')
    print('Spam' if le.inverse_transform(new_email_prediction)[0] == 'spam' else 'Not Spam')
else:
    print(f"Error: The file at {file_path} was not found.")
