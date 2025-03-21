import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Suppress Pandas FutureWarning for downcasting
pd.set_option('future.no_silent_downcasting', True)

# Load dataset
data = pd.read_csv('Kidney_data.csv')

# Ensure 'saved_model' directory exists
os.makedirs('saved_model', exist_ok=True)

# Replace '?' with NaN and convert numeric columns
data.replace('?', pd.NA, inplace=True)

# Define feature names
feature_names = ['age', 'bp', 'sg', 'al', 'bgr', 'bu', 'sc', 'hemo', 'pcv', 'wc']

# Ensure columns exist in the dataset and rename 'wc' if necessary
data = data.rename(columns={'wc': 'wc'})  # No change since 'wc' matches
numeric_cols = [col for col in feature_names if col in data.columns]

# Ensure numeric columns are numeric
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Handle missing values with median
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Clean and encode the 'classification' column (binary: 0 = no CKD, 1 = CKD)
if 'classification' in data.columns:
    data['classification'] = data['classification'].astype(str).str.strip().str.lower()
    data['classification'] = data['classification'].replace({'notckd': 0, 'ckd': 1})
    if data['classification'].isna().any() or not all(data['classification'].isin([0, 1, 'nan'])):
        data['classification'] = data['classification'].fillna(0)  # Treat NaN as notckd
        data.loc[~data['classification'].isin([0, 1]), 'classification'] = 0  # Force invalid to 0
else:
    print("Error: 'classification' column not found in dataset.")
    exit(1)

# Define features and target
if 'id' in data.columns:
    X = data.drop(columns=['id', 'classification'])
else:
    X = data.drop(columns=['classification'])
y = data['classification']

# Select only the specified features
X = X[feature_names]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
with open('saved_model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save preprocessed data
pd.DataFrame(X_scaled, columns=feature_names).to_csv('saved_model/processed_features.csv', index=False)
y.to_csv('saved_model/processed_target.csv', index=False)

print("Data preprocessing completed. Scaler and preprocessed data saved to 'saved_model/'.")