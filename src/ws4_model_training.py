# ==========================================
# WS4 - Advanced Model Development (SMOTE + XGBoost + Tuning)
# ==========================================
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

input_file = "data/processed/credit_features.csv"
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Error: {input_file} missing. Run WS3 first.")

df = pd.read_csv(input_file)
print("--- WS4: Advanced Credit Risk Training ---")

# 1. Define Features & Target
X = df.drop('Risk', axis=1)
y = df['Risk']

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')

# 4. Apply SMOTE (Synthetic Minority Over-sampling)
# This invents realistic 'Bad Loan' profiles until we have a perfect 50/50 split
print("[Data Engineering] Applying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f" -> Old training target count:\n{y_train.value_counts().to_string()}")
print(f" -> New training target count:\n{y_train_resampled.value_counts().to_string()}\n")

# 5. Define Models & Hyperparameter Grids
# We provide a 'grid' of options, and the system will test them all to find the best
models_and_grids = {
    "LogisticRegression": (
        LogisticRegression(random_state=42, max_iter=1000),
        {'C': [0.1, 1.0, 10.0]} # Regularization strength
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [5, 10]} # Number of trees and depth
    ),
    "XGBoost": (
        XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=5),
        {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]} 
    )
}

# 6. Train using GridSearchCV
print("[Training Models with Hyperparameter Tuning]")
for name, (model, grid) in models_and_grids.items():
    print(f" -> Tuning & Training {name}... (this may take a few seconds)")
    
    # GridSearchCV tests every combination in the grid using cross-validation
    # We set scoring='recall' so it specifically looks for combinations that catch bad loans
    search = GridSearchCV(estimator=model, param_grid=grid, scoring='recall', cv=3, n_jobs=-1)
    search.fit(X_train_resampled, y_train_resampled)
    
    # Extract the winning configuration
    best_model = search.best_estimator_
    
    # Save the optimized model
    model_path = f"models/{name}_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"    Best Params: {search.best_params_}")
    print(f"    Saved optimized model to {model_path}\n")

# 7. Save test data for WS5
joblib.dump((X_test_scaled, y_test), 'data/processed/test_data.pkl')

print("==========================================")
print(" WS4 Complete. Optimized models saved.")
print("==========================================")