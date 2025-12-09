# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import joblib
import numpy as np

# --- Configuration ---
# NOTE: Ensure 'heart_disease.csv' is in the same directory as this file.
csv_path = 'heart_disease.csv'

# --- 1. Load Data ---
data = pd.read_csv(csv_path)

# --- 2. Handle Missing Values (Imputation) ---
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode().iloc[0])
    else:
        # Impute numeric missing values with the mean
        data[col] = data[col].fillna(data[col].mean())

# --- 3. Encode Categorical Columns ---
# Save the final cleaned/encoded data for unsupervised tasks
df_encoded = data.copy()
# Map target variable
df_encoded['Heart Disease Status'] = df_encoded['Heart Disease Status'].map({'No': 0, 'Yes': 1})

# Encoding feature columns
label_enc = LabelEncoder()
categorical_cols = [col for col in df_encoded.columns if df_encoded[col].dtype == 'object']
for col in categorical_cols:
    df_encoded[col] = label_enc.fit_transform(df_encoded[col])

# --- CLASSIFICATION PIPELINE ---

# 4. Separate features and target
X = df_encoded.drop("Heart Disease Status", axis=1)
y = df_encoded["Heart Disease Status"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Identify Numerical Features for Scaling
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

# 7. Scale Numeric Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

# Re-assemble dataframes for Logistic Regression
X_train_log = X_train.copy()
X_train_log[numerical_features] = X_train_scaled

X_test_log = X_test.copy()
X_test_log[numerical_features] = X_test_scaled

# 8. Train Logistic Regression
log_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
log_model.fit(X_train_log, y_train)
log_accuracy = accuracy_score(y_test, log_model.predict(X_test_log))

# 9. Train Random Forest (No Scaling needed for RF)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

# --- UNSUPERVISED PIPELINE ---

# 10. K-Means Clustering
X_scaled_all = X.copy()
X_scaled_all[numerical_features] = scaler.transform(X_scaled_all[numerical_features]) 
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # 3 clusters for Low, Medium, High Risk
kmeans.fit(X_scaled_all)

# Extract cluster centers (prototypes of each risk group)
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

# 11. Pattern Mining (Apriori)
# Binarize data (requires boolean or 0/1 columns for Apriori)
temp_df = df_encoded.copy()
temp_df['High_Cholesterol'] = np.where(temp_df['Cholesterol Level'] > 240, 1, 0)
temp_df['Old_Age'] = np.where(temp_df['Age'] > 60, 1, 0)
temp_df = temp_df.select_dtypes(include=[np.number]).apply(lambda x: x.astype(bool)) 
# Drop complex continuous features used for clustering/classification but not ideal for Apriori rules
temp_df = temp_df.drop(columns=['Age', 'Blood Pressure', 'BMI', 'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level', 'Cholesterol Level', 'Sleep Hours'], errors='ignore')

# Apriori algorithm
frequent_itemsets = apriori(temp_df, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
top_rules = rules.sort_values('lift', ascending=False).head(5)


# --- 12. Save All Assets ---
joblib.dump(rf_model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(log_model, 'logistic_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')
joblib.dump(cluster_centers, 'cluster_centers.pkl')
joblib.dump(top_rules, 'association_rules.pkl') 

print("="*50)
print(f"✅ Training and Analysis Complete. All Assets Saved.")
print(f"  Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"  Logistic Regression Accuracy: {log_accuracy:.4f}")
print("="*50)