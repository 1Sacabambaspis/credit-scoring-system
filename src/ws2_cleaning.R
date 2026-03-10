# ==========================================
# WS2 - Data Preparation & Cleaning (R)
# ==========================================

# 1. Load Data
file_path <- "data/raw/german_credit.csv"
if (!file.exists(file_path)) {
  stop("Error: german_credit.csv not found in data/raw/. Please check the file path.")
}
df <- read.csv(file_path, stringsAsFactors = FALSE)

# 2. Remove Duplicate Records
df <- unique(df)

# 3. Recode Risk Target Variable
# [cite_start]The dataset defines Good = 1, Bad = 2[cite: 946]. 
# For standard modeling, we recode: Good credit -> 0, Bad credit -> 1.
df$Risk <- ifelse(df$Risk == 2, 1, 0)

# 4. Outlier Treatment (IQR Capping)
# Detect and cap extreme values for numerical attributes
cap_outliers <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  upper <- q3 + 1.5 * iqr
  lower <- q1 - 1.5 * iqr
  x[x > upper] <- upper
  x[x < lower] <- lower
  return(x)
}

num_cols <- c("Credit_Amount", "Duration_Months", "Installment_Rate", "Age")
for (col in num_cols) {
  df[[col]] <- cap_outliers(df[[col]])
}

# 5. Handle Missing Values
df <- na.omit(df)

# 6. Encode Ordinal Variables (Ordered Integers)
# [cite_start]Checking Account: A14 (None) to A13 (>= 200 DM) [cite: 936, 937]
df$Checking_Account <- as.integer(factor(df$Checking_Account, 
                                         levels = c("A14", "A11", "A12", "A13")))

# [cite_start]Savings Account: A65 (None) to A64 (>= 1000 DM) [cite: 940]
df$Savings_Account <- as.integer(factor(df$Savings_Account, 
                                        levels = c("A65", "A61", "A62", "A63", "A64")))

# [cite_start]Employment Since: A71 (Unemployed) to A75 (>= 7 years) [cite: 940, 941]
df$Employment_Since <- as.integer(factor(df$Employment_Since, 
                                         levels = c("A71", "A72", "A73", "A74", "A75")))

# 7. One-Hot Encoding for Nominal Variables
nominal_cols <- c("Credit_History", "Purpose", "Personal_Status_Sex", 
                  "Other_Debtors", "Property", "Other_Installment_Plans", 
                  "Housing", "Job", "Telephone", "Foreign_Worker")

# Convert nominal columns to factors
for (col in nominal_cols) {
  df[[col]] <- as.factor(df[[col]])
}

# Create a one-hot encoded matrix (omitting the intercept)
dummies <- model.matrix(~ . - 1, data = df[nominal_cols])

# Combine numerical/ordinal columns with the new dummy variables
df_clean <- cbind(df[!(names(df) %in% nominal_cols)], dummies)

# 8. Export Processed Data
write.csv(df_clean, "data/processed/credit_clean.csv", row.names = FALSE)

cat("\n==========================================\n")
cat(" WS2 Complete. Cleaned data saved to:\n")
cat(" -> data/processed/credit_clean.csv\n")
cat("==========================================\n")