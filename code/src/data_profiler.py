import pandas as pd
import math
from datetime import datetime
from sklearn.ensemble import IsolationForest
import streamlit as st

# Load the dataset
data = pd.read_csv("customer.csv")

# 1. Rule: Transaction Amount should match Reported Amount (allow 1% deviation for cross-currency transactions)
def validate_transaction_amount(row):
    if abs(row["Transaction_Amount"] - row["Reported_Amount"]) > (0.01 * row["Reported_Amount"]):
        return "Mismatch in Transaction Amount"
    return "Valid"

# 2. Rule: Account Balance must not be negative (except marked OD accounts)
def validate_account_balance(row):
    if row["Account_Balance"] < 0 and "OD" not in row:
        return "Negative Account Balance"
    return "Valid"

# 3. Rule: Transaction Date validations
def validate_transaction_date(row):
    today = datetime.now()
    txn_date = datetime.strptime(row["Transaction_Date"], "%Y-%m-%d")
    if txn_date > today:
        return "Future Transaction Date"
    if (today - txn_date).days > 365:
        return "Transaction older than 365 days"
    return "Valid"

# 4. Rule: High-risk transactions
def calculate_risk_score(row):
    risk_score = 0
    if row["Transaction_Amount"] > 5000:
        risk_score += 2
    if row["Country"] in ["High-Risk-Country1", "High-Risk-Country2"]:  # Define high-risk countries
        risk_score += 3
    if row["Transaction_Amount"] % 1000 == 0:  # Round-number check
        risk_score += 1
    return risk_score

# Apply the validations
data["Transaction_Validation"] = data.apply(validate_transaction_amount, axis=1)
data["Account_Validation"] = data.apply(validate_account_balance, axis=1)
data["Date_Validation"] = data.apply(validate_transaction_date, axis=1)
data["Risk_Score"] = data.apply(calculate_risk_score, axis=1)
model = IsolationForest(contamination=0.1)
data["Anomaly_Flag"] = model.fit_predict(data[["Account_Balance", "Transaction_Amount"]])

# Output results
print(data[["Customer_ID", "Transaction_Validation", "Account_Validation", "Date_Validation", "Risk_Score", "Anomaly_Flag"]])

# Export the flagged results
data.to_csv("validated_data.csv", index=False)
#st.write(data[["Customer_ID", "Transaction_Validation", "Account_Validation", "Date_Validation", "Risk_Score", "Anomaly_Flag"]])