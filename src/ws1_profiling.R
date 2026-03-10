# ==========================================
# WS1 - Loan Data Profiling
# ==========================================

# 1. Load the dataset
file_path <- "data/raw/german_credit.csv"

if (!file.exists(file_path)) {
  stop("Error: german_credit.csv not found. Please check the file path.")
}

credit_data <- read.csv(file_path, stringsAsFactors = TRUE)

# 2. Define Target and Sensitive Attributes
target_variable <- "Risk" 
sensitive_attributes <- c("Age", "Personal_Status_Sex")

cat("\n==========================================\n")
cat("          LOAN DATASET PROFILE              \n")
cat("==========================================\n")

cat("\n[1] TARGET VARIABLE:\n")
cat(" ->", target_variable, "\n")

cat("\n[2] SENSITIVE ATTRIBUTES DEFINED:\n")
cat(" ->", paste(sensitive_attributes, collapse=", "), "\n")

# 3. Categorize Applicant Attributes (Data Types)
cat("\n[3] ATTRIBUTE CATEGORIZATION (Data Types):\n")
str(credit_data)

# 4. Produce Loan Dataset Profile (Summary Statistics)
cat("\n[4] DATASET PROFILING SUMMARY:\n")
summary(credit_data)

cat("\n==========================================\n")
cat(" WS1 Profiling Complete. Ready for WS2.   \n")
cat("==========================================\n")