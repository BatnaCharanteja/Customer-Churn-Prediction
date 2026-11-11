# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their service using machine learning. Built with Python, scikit-learn, and pandas.

## What is this project about?

Customer churn means a customer stops using your service. For a telecom company, that's someone cancelling their phone or internet plan. It costs way more to get a new customer than to keep an existing one, so if we can predict who's about to leave, the company can try to retain them with offers or better support.

I used a dataset of ~7,000 telecom customers with info about their demographics, services, contracts, and billing. The goal is to build a model that can look at a customer's profile and predict whether they'll churn or not.

## How the pipeline works

1. **Data loading and exploration** - Loaded the dataset and checked the churn distribution. Turns out about 27% of customers churned, so the dataset is imbalanced (more "no" than "yes"). Also plotted some charts to understand patterns.

2. **Data cleaning** - Found that `TotalCharges` had some blank values stored as strings (for brand new customers). Converted those to numeric and dropped the 11 affected rows. Also removed `customerID` since it's just a unique ID with no predictive power.

3. **Preprocessing** - One-hot encoded all the categorical columns (like Contract, InternetService, etc). Used `drop_first=True` to avoid the dummy variable trap. Split into 80/20 train/test with stratification, then applied StandardScaler. Made sure to fit the scaler on training data only to avoid data leakage.

4. **Logistic Regression** - Trained this as a baseline. It's simple and interpretable. Used `class_weight='balanced'` to handle the class imbalance.

5. **Random Forest** - Trained this for comparison. It can capture non-linear patterns that logistic regression misses. Also used balanced class weights.

6. **Evaluation** - Compared both models on accuracy, precision, recall, and F1-score. Didn't rely on accuracy alone because with 73% of customers not churning, a model that always predicts "no churn" would get 73% accuracy while being totally useless.

7. **Feature importance** - Looked at which features matter most for predicting churn. Contract type, tenure, and charges came out on top.

## Results

| Metric | Logistic Regression | Random Forest |
|--------|:---:|:---:|
| Accuracy | 0.6908 | 0.6894 |
| Precision | 0.4632 | 0.4591 |
| Recall | 0.7084 | 0.6598 |
| F1-Score | 0.5602 | 0.5414 |

I went with **Logistic Regression** as the final model. It performed slightly better across all metrics, and it's simpler and easier to explain. When two models perform similarly, the simpler one is usually the better choice.

## Key findings

- **Contract type matters the most.** Month-to-month customers churn at ~42% vs ~8% for two-year contracts. If I were advising the business, I'd say push contract upgrades with discounts.
- **New customers are high risk.** Most churn happens in the first 12 months. There should be some kind of onboarding/welcome program.
- **Higher monthly charges = more churn.** Customers paying $70-80/month leave more than those paying $25-30. Might need better value proposition for high-spend customers.

## Why I made certain decisions

**Why classification and not regression?**
The target (Churn) is Yes/No - it's categorical. We're predicting a class label, not a number.

**Why one-hot encoding instead of label encoding?**
Label encoding puts numbers like 0, 1, 2 on categories, which implies an order that doesn't exist. "DSL" isn't less than "Fiber optic". One-hot encoding treats each category independently.

**Why not just use accuracy?**
Because the data is imbalanced. A model that always says "no churn" gets 73% accuracy but catches zero churners - completely useless for a business that wants to retain customers.

**Which is more important - precision or recall?**
For churn, recall matters more. A false negative (missing a churner) means we lose the customer. A false positive (flagging someone who wasn't going to leave) just means we offered them a retention deal unnecessarily - that's a much cheaper mistake.

**Why Logistic Regression vs Random Forest?**
It's a standard simple-vs-complex comparison. Logistic Regression is linear and fast, Random Forest is an ensemble that captures non-linear patterns. If the simple model works well enough, there's no need for complexity.

## What I'd improve with more time

- Try XGBoost or LightGBM (usually outperform Random Forest)
- Do hyperparameter tuning with GridSearchCV
- Try SMOTE for handling class imbalance at the data level
- Use cross-validation instead of a single train/test split
- Tune the classification threshold (currently 0.5 by default)
- Build a simple Flask API for predictions

## How to run

```
pip install pandas numpy scikit-learn matplotlib seaborn
python generate_dataset.py
python churn_prediction.py
```

The script will print all the results and save 7 visualization charts as PNG files.

## Files

- `generate_dataset.py` - Creates a synthetic dataset that matches the Kaggle Telco Churn format
- `churn_prediction.py` - The main ML pipeline
- `README.md` - You're reading it
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - The dataset (generated)
- `viz_*.png` - Visualization charts (generated after running the pipeline)
- `results_summary.txt` - Final model scores
