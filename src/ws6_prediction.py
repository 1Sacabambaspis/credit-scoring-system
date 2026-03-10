# ==========================================
# WS6 - Loan Risk Prediction & Approval Grouping
# ==========================================
import pandas as pd
import joblib
import os

print("--- WS6: Final Loan Decision Engine ---")

# 1. Identify and load the winning model
with open("models/best_model_name.txt", "r") as f:
    best_model_name = f.read().strip()

print(f"[System] Loading best model: {best_model_name}")
model = joblib.load(f"models/{best_model_name}_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# 2. Load the full cleaned dataset (simulating new applicants)
df_features = pd.read_csv("data/processed/credit_features.csv")
df_raw = pd.read_csv("data/processed/credit_clean.csv") # To attach predictions to raw data

# We drop the actual 'Risk' target to simulate predicting on unseen data
X_new = df_features.drop('Risk', axis=1)

# 3. Scale the data exactly as we did in training
X_new_scaled = scaler.transform(X_new)

# 4. Generate Probabilities instead of hard 1/0 predictions
# predict_proba returns an array: [Probability_Good(0), Probability_Bad(1)]
# We extract the second column (index 1) to get the Bad Loan probability
probabilities = model.predict_proba(X_new_scaled)[:, 1]

# 5. Apply Business Logic & Risk Grouping
# To compensate for the model's low recall, we lower the risk threshold.
# If the model thinks there is even a 35% chance of default, we reject it.
THRESHOLD = 0.35 

decisions = []
risk_groups = []

for prob in probabilities:
    if prob >= THRESHOLD:
        risk_groups.append("High Risk")
        decisions.append("Rejected")
    else:
        risk_groups.append("Low Risk")
        decisions.append("Approved")

# 6. Construct the Final Output Payload
df_final = df_raw.copy()
df_final['Default_Probability'] = [round(p, 3) for p in probabilities]
df_final['Risk_Group'] = risk_groups
df_final['Loan_Decision'] = decisions

output_file = "data/processed/predictions.csv"
df_final.to_csv(output_file, index=False)

print("\n[Business Logic Applied]")
print(f" -> Rejection Threshold set to: {THRESHOLD * 100}% probability")
print("\nFinal Decision Distribution:")
print(df_final['Loan_Decision'].value_counts())

print("\n==========================================")
print(f" PIPELINE COMPLETE! Final output saved to:\n -> {output_file}")
print("==========================================")