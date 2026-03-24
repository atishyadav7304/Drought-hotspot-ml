# ==============================
# Drought Hotspot ML Pipeline
# ==============================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==============================
# STEP 1: Load Data
# ==============================

# Replace with your CSV file path
data = pd.read_csv("ML_Data.csv")

print("Dataset Preview:")
print(data.head())

# ==============================
# STEP 2: Data Cleaning
# ==============================

# Drop geometry column if exists
if 'geometry' in data.columns:
    data = data.drop(columns=['geometry'])

# Drop missing values
data = data.dropna()

# ==============================
# STEP 3: Feature Selection
# ==============================

# Example columns (modify based on your CSV)
features = ['NDVI', 'precipitation']

X = data[features]

# ==============================
# STEP 4: Normalization
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# STEP 5: Unsupervised Learning (KMeans)
# ==============================

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster counts:")
print(data['Cluster'].value_counts())

# ==============================
# STEP 6: Label Creation (IMPORTANT)
# ==============================

# Convert clusters into drought labels
# Example logic (adjust after visualization)

def label_drought(cluster):
    if cluster == 0:
        return 0   # Low drought
    elif cluster == 1:
        return 1   # Moderate
    else:
        return 2   # High drought

data['Drought_Class'] = data['Cluster'].apply(label_drought)

# ==============================
# STEP 7: Supervised Learning
# ==============================

y = data['Drought_Class']

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prediction
y_pred = rf_model.predict(X_test)

# ==============================
# STEP 8: Evaluation
# ==============================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# STEP 9: Feature Importance
# ==============================

importance = rf_model.feature_importances_

for i, col in enumerate(features):
    print(f"{col}: {importance[i]:.4f}")

# Plot importance
plt.bar(features, importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

import os

# ==============================
# STEP 10: Save Results
# ==============================

# Save full dataset with predictions
data['Predicted_Class'] = rf_model.predict(X_scaled)

data.to_csv(r"C:\Users\atish\Documents\M.Tech_Land_and_Water_Resource_Engineering\2nd Semester M.Tech\CE60252 Statistics and Machine Learning for Water Resource Engineering\Drought Hazard ML Project\Drought_Hotspot_Result_1.csv", index=False)
data[['Longitude','Latitude']] = data['.geo'].str.extract(r'\[([0-9\.]+),([0-9\.]+)\]').astype(float)
data.drop(columns=['.geo'], errors='ignore').to_csv(r"C:\Users\atish\Documents\M.Tech_Land_and_Water_Resource_Engineering\2nd Semester M.Tech\CE60252 Statistics and Machine Learning for Water Resource Engineering\Drought Hazard ML Project\Drought_Hotspot_Result_1.csv", index=False)
print("\nResults saved as Drought_Hotspot_Result.csv")
