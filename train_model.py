import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load preprocessed data
try:
    X = pd.read_csv('saved_model/processed_features.csv')
    y = pd.read_csv('saved_model/processed_target.csv')
    print(f"Loaded {len(X)} samples with {X.shape[1]} features.")
except FileNotFoundError:
    print("Error: Preprocessed files ('processed_features.csv' or 'processed_target.csv') not found.")
    exit(1)

# Define expected features
feature_names = ['age', 'bp', 'sg', 'al', 'bgr', 'bu', 'sc', 'hemo', 'pcv', 'wc']
if list(X.columns) != feature_names:
    print("Error: Feature names in processed_features.csv do not match expected order.")
    print("Expected:", feature_names)
    print("Found:", list(X.columns))
    exit(1)

# Check target encoding and ensure binary
if len(y['classification'].unique()) != 2 or not all(val in [0, 1] for val in y['classification'].unique()):
    print("Error: Target must be binary (0 and 1). Found:", y['classification'].unique())
    exit(1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not CKD', 'CKD']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
with open('saved_model/ckd_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training completed and saved to 'saved_model/ckd_model.pkl'.")