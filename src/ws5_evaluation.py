# ==========================================
# WS5 - Model Evaluation & Selection
# ==========================================
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

print("--- WS5: Credit Model Evaluation ---")

# 1. Load the hold-out test data
test_data_path = 'data/processed/test_data.pkl'
if not os.path.exists(test_data_path):
    raise FileNotFoundError("Error: Test data missing. Run WS4 first.")

X_test_scaled, y_test = joblib.load(test_data_path)

# 2. Define the models to evaluate
model_names = ["LogisticRegression", "RandomForest", "XGBoost"]
results = []

# 3. Evaluate each model
for name in model_names:
    model_path = f"models/{name}_model.pkl"
    if not os.path.exists(model_path):
        print(f"Skipping {name}, model file not found.")
        continue
        
    # Load the trained model into memory
    model = joblib.load(model_path)
    
    # Predict the Risk on the test data (0 = Good, 1 = Bad)
    y_pred = model.predict(X_test_scaled)
    
    # Compute metrics specifically for Class 1 (Bad Loans)
    # Missing a bad loan is costly, so Recall is our VIP metric
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    # Store results
    results.append({
        "Model": name,
        "Precision (Bad Loans)": round(precision, 3),
        "Recall (Bad Loans)": round(recall, 3),
        "F1-Score": round(f1, 3)
    })
    
    # Print individual confusion matrices
    print(f"\n[{name}] Confusion Matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        columns=['Predicted Good(0)', 'Predicted Bad(1)'],
        index=['Actual Good(0)', 'Actual Bad(1)']
    ))

# 4. Create and display the comparison table
results_df = pd.DataFrame(results)
print("\n--- Model Comparison Table ---")
print(results_df.to_string(index=False))

# 5. Select the Final Scoring Model
# Prioritize Highest Recall. If tied, pick the one with the better F1-Score
best_model_row = results_df.sort_values(by=['Recall (Bad Loans)', 'F1-Score'], ascending=[False, False]).iloc[0]
best_model_name = best_model_row['Model']

print("\n==========================================")
print(f" FINAL SELECTION: {best_model_name}")
print(f" Reason: Highest recall for detecting bad loans.")
print("==========================================")

# Save a marker so WS6 knows which model won
with open("models/best_model_name.txt", "w") as f:
    f.write(best_model_name)