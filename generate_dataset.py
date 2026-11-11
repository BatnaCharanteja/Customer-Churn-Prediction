# Generates a synthetic dataset that matches the Kaggle Telco Customer Churn dataset
# Same columns, same distributions, same quirks (like blank TotalCharges values)
# Run this once before running churn_prediction.py

import numpy as np
import pandas as pd

np.random.seed(42)

N = 7043  # same size as the real kaggle dataset

# helper function
def pick(options, weights, size):
    return np.random.choice(options, size=size, p=weights)

# customer IDs
customer_ids = [f"{i:04d}-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for i in range(1, N + 1)]

# demographics
gender = pick(["Male", "Female"], [0.505, 0.495], N)
senior_citizen = pick([0, 1], [0.84, 0.16], N)
partner = pick(["Yes", "No"], [0.48, 0.52], N)
dependents = pick(["Yes", "No"], [0.30, 0.70], N)

# tenure - bimodal distribution (lots of new and lots of loyal customers)
tenure = np.concatenate([
    np.random.randint(1, 12, size=int(N * 0.35)),
    np.random.randint(12, 48, size=int(N * 0.30)),
    np.random.randint(48, 73, size=N - int(N * 0.35) - int(N * 0.30))
])
np.random.shuffle(tenure)

# services
phone_service = pick(["Yes", "No"], [0.90, 0.10], N)
multiple_lines = np.where(
    phone_service == "No", "No phone service",
    pick(["Yes", "No"], [0.42, 0.58], N)
)

internet_service = pick(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22], N)

# these services depend on whether customer has internet
def inet_service(yes_prob=0.35):
    result = []
    for inet in internet_service:
        if inet == "No":
            result.append("No internet service")
        else:
            result.append(np.random.choice(["Yes", "No"], p=[yes_prob, 1 - yes_prob]))
    return np.array(result)

online_security = inet_service(0.35)
online_backup = inet_service(0.40)
device_protection = inet_service(0.40)
tech_support = inet_service(0.35)
streaming_tv = inet_service(0.45)
streaming_movies = inet_service(0.45)

# contract and billing
contract = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24], N)
paperless_billing = pick(["Yes", "No"], [0.59, 0.41], N)
payment_method = pick(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    [0.34, 0.23, 0.22, 0.21], N
)

# charges
monthly_charges = np.round(
    20
    + (internet_service == "DSL") * np.random.uniform(10, 20, N)
    + (internet_service == "Fiber optic") * np.random.uniform(25, 40, N)
    + (phone_service == "Yes") * np.random.uniform(5, 15, N)
    + (online_security == "Yes") * np.random.uniform(5, 10, N)
    + (tech_support == "Yes") * np.random.uniform(5, 10, N)
    + (streaming_tv == "Yes") * np.random.uniform(5, 12, N)
    + (streaming_movies == "Yes") * np.random.uniform(5, 12, N)
    + np.random.normal(0, 3, N),
    2
)
monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

total_charges = np.round(tenure * monthly_charges * np.random.uniform(0.9, 1.1, N), 2)

# add some missing TotalCharges (blank strings, like in the real dataset)
missing_idx = np.random.choice(range(N), size=11, replace=False)
total_charges_str = total_charges.astype(str)
total_charges_str[missing_idx] = " "
tenure[missing_idx] = 0

# churn - based on realistic factors, not random
churn_prob = np.full(N, 0.15)

# contract type is the biggest factor
churn_prob += (contract == "Month-to-month") * 0.25
churn_prob -= (contract == "Two year") * 0.12
churn_prob -= (contract == "One year") * 0.06

# fiber optic without support = frustrated customers
churn_prob += ((internet_service == "Fiber optic") & (online_security == "No")) * 0.10
churn_prob += ((internet_service == "Fiber optic") & (tech_support == "No")) * 0.08

# tenure effect
churn_prob -= (tenure > 48) * 0.15
churn_prob -= (tenure > 24) * 0.05
churn_prob += (tenure < 6) * 0.08

# other factors
churn_prob += (payment_method == "Electronic check") * 0.05
churn_prob += (senior_citizen == 1) * 0.04
churn_prob += (monthly_charges > 80) * 0.05

churn_prob = np.clip(churn_prob, 0.02, 0.85)
churn = np.array(["Yes" if np.random.random() < p else "No" for p in churn_prob])

# build dataframe
df = pd.DataFrame({
    "customerID": customer_ids,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges_str,
    "Churn": churn,
})

# save
output = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df.to_csv(output, index=False)

churn_rate = (churn == "Yes").mean() * 100
print(f"Dataset saved: {output}")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"Churn rate: {churn_rate:.1f}%")
print(f"Missing TotalCharges: {(total_charges_str == ' ').sum()}")
