import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

print("1. Loading Combined Weather Dataset...")
# Read the dataset
df = pd.read_csv("Final_Weather_Dataset.csv")  # Ensure this file exists in the working directory

# 2. Feature Engineering
print("2. Engineering features (extracting month)...")
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# Create binary Rain target (1 if Rain_mm > 0, else 0)
df['Is_Rain'] = (df['Rain_mm'] > 0).astype(int)

# Drop missing values in our key columns
df = df.dropna(subset=['City', 'Month', 'Sunshine_Seconds', 'WindSpeed_Max', 'Temp_Mean', 'Is_Rain'])

# Define features and targets
X = df[['City', 'Month', 'Sunshine_Seconds', 'WindSpeed_Max']]
y_temp = df['Temp_Mean']
y_rain = df['Is_Rain']

# 3. Preprocessing Setup
# One-hot encode City, keep numeric features scaling-ready
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), ['City']),
        ('num', StandardScaler(), ['Month', 'Sunshine_Seconds', 'WindSpeed_Max'])
    ])

print("3. Preprocessing data and splitting datasets...")
X_processed = preprocessor.fit_transform(X)

# Split for Temperature Model
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_processed, y_temp, test_size=0.2, random_state=42)
# Split for Rain Model
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_processed, y_rain, test_size=0.2, random_state=42)

# 4. Training Temperature Model (Regressor)
print("4. Training Temperature Model (RandomForestRegressor)...")
temp_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
temp_model.fit(X_train_t, y_train_t)
temp_preds = temp_model.predict(X_test_t)
print(f"   -> Temp Model Mean Absolute Error: {mean_absolute_error(y_test_t, temp_preds):.2f}°C")

# 5. Training Rain Model (Classifier)
print("5. Training Rain Model (RandomForestClassifier)...")
rain_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
rain_model.fit(X_train_r, y_train_r)
rain_preds = rain_model.predict(X_test_r)
print(f"   -> Rain Model Accuracy: {accuracy_score(y_test_r, rain_preds) * 100:.2f}%")

# 6. Saving Everything
print("6. Saving models and preprocessor pipeline...")
with open("temp_model.pkl", "wb") as f:
    pickle.dump(temp_model, f)

with open("rain_model.pkl", "wb") as f:
    pickle.dump(rain_model, f)

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("\n🎉 Success! Created temp_model.pkl, rain_model.pkl, and preprocessor.pkl.")
print("You can now run 'streamlit run app.py'")