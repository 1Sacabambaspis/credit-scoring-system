# ==========================================
# WS3 - Exploratory Analysis & Feature Design
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the cleaned dataset from WS2
input_file = "data/processed/credit_clean.csv"
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Error: {input_file} missing. Run WS2 R script first.")

df = pd.read_csv(input_file)
print("--- WS3: Loan Exploratory Analysis & Feature Design ---")
print(f"Loaded clean dataset with {df.shape[0]} records and {df.shape[1]} columns.\n")

# 2. Exploratory Data Analysis (EDA)
print("[EDA] Default Risk Distribution:")
risk_counts = df['Risk'].value_counts(normalize=True) * 100
print(risk_counts.to_string())

# Optional: Generate a quick plot to visualize relationships
# (Saves to the notebooks folder to keep the root clean)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Risk', y='Credit_Amount', data=df)
plt.title('Credit Amount vs. Default Risk')
plt.savefig('notebooks/credit_vs_risk_plot.png')
print("\n[EDA] Saved 'credit_vs_risk_plot.png' to notebooks/ folder.")

# 3. Feature Construction (Domain-Specific)
print("\n[Feature Engineering] Constructing new loan attributes...")

# Feature A: Credit amount requested per month of the loan
df['credit_per_month'] = df['Credit_Amount'] / df['Duration_Months']

# Feature B: Categorize loan duration into risk groups
def categorize_duration(months):
    if months <= 12:
        return 'short'
    elif months <= 36:
        return 'medium'
    else:
        return 'long'

df['duration_group'] = df['Duration_Months'].apply(categorize_duration)

# One-hot encode the new duration_group so it is machine-readable
duration_dummies = pd.get_dummies(df['duration_group'], prefix='duration', drop_first=True)
df = pd.concat([df, duration_dummies], axis=1)

# Drop the text column since we encoded it
df = df.drop('duration_group', axis=1)

# 4. Final Feature List Definition
output_file = "data/processed/credit_features.csv"
df.to_csv(output_file, index=False)

print("\n==========================================")
print(f" WS3 Complete. Final modeling features saved to:\n -> {output_file}")
print("==========================================")