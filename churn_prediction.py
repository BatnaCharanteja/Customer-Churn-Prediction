# Customer Churn Prediction
# Using Logistic Regression and Random Forest to predict which telecom customers will leave
# Dataset: Telco Customer Churn (7043 customers, 21 features)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 120


# ============================
# 1. Load and explore the data
# ============================

print("STEP 1: Loading data...")

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())
print()
print(df.dtypes)
print()
print(df.describe())

# check how many customers churned vs didn't
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100
print(f"\nChurn distribution:")
print(f"  No:  {churn_counts['No']} ({churn_pct['No']:.1f}%)")
print(f"  Yes: {churn_counts['Yes']} ({churn_pct['Yes']:.1f}%)")
# so its imbalanced - about 73/27 split. need to keep this in mind later

# --- Visualization 1: churn by contract type ---
# I want to see if contract type affects churn because intuitively
# month-to-month customers should leave more easily

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

contract_churn = pd.crosstab(df['Contract'], df['Churn'])
contract_churn.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'], edgecolor='white')
axes[0].set_title('Churn Count by Contract Type', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Contract Type')
axes[0].set_ylabel('Number of Customers')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].legend(['No Churn', 'Churned'], loc='upper right')

# churn rate is more useful than raw count here
churn_rate_by_contract = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).sort_values(ascending=False)

axes[1].barh(churn_rate_by_contract.index, churn_rate_by_contract.values, 
             color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='white', height=0.5)
axes[1].set_title('Churn Rate (%) by Contract Type', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Churn Rate (%)')
for i, v in enumerate(churn_rate_by_contract.values):
    axes[1].text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('viz_churn_by_contract.png', bbox_inches='tight')
plt.close()
print("Saved: viz_churn_by_contract.png")

# --- Visualization 2: churn by tenure ---
# want to see when customers are most likely to leave

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist([df[df['Churn'] == 'No']['tenure'], df[df['Churn'] == 'Yes']['tenure']], 
             bins=30, color=['#2ecc71', '#e74c3c'], label=['No Churn', 'Churned'],
             alpha=0.75, edgecolor='white', stacked=False)
axes[0].set_title('Tenure Distribution by Churn Status', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Tenure (months)')
axes[0].set_ylabel('Number of Customers')
axes[0].legend()

sns.boxplot(x='Churn', y='tenure', data=df, ax=axes[1],
            palette={'No': '#2ecc71', 'Yes': '#e74c3c'}, width=0.4)
axes[1].set_title('Tenure by Churn Status (Box Plot)', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Churned?')
axes[1].set_ylabel('Tenure (months)')

plt.tight_layout()
plt.savefig('viz_churn_by_tenure.png', bbox_inches='tight')
plt.close()
print("Saved: viz_churn_by_tenure.png")

# --- Visualization 3: monthly charges ---

fig, ax = plt.subplots(figsize=(10, 5))
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True,
            palette={'No': '#2ecc71', 'Yes': '#e74c3c'}, alpha=0.4, ax=ax)
ax.set_title('Monthly Charges Distribution by Churn Status', fontweight='bold', fontsize=13)
ax.set_xlabel('Monthly Charges ($)')
ax.set_ylabel('Density')
plt.tight_layout()
plt.savefig('viz_charges_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: viz_charges_distribution.png")

# --- Visualization 4: correlation heatmap ---
# only works for numeric columns but still useful to see relationships

df_corr = df.copy()
df_corr['Churn_numeric'] = (df_corr['Churn'] == 'Yes').astype(int)

numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
if 'Churn_numeric' not in numeric_cols:
    numeric_cols.append('Churn_numeric')

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df_corr[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Correlation Heatmap (Numeric Features)', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('viz_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved: viz_correlation_heatmap.png")


# ============================
# 2. Data Cleaning
# ============================

print("\nSTEP 2: Cleaning data...")

# TotalCharges has some blank values stored as strings which causes issues
# found this by noticing the dtype was 'object' instead of float
print(f"TotalCharges dtype before: {df['TotalCharges'].dtype}")
print(f"Blank values: {(df['TotalCharges'] == ' ').sum()}")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# only 11 rows have missing TotalCharges, safe to just drop them
rows_before = len(df)
df = df.dropna(subset=['TotalCharges'])
print(f"Dropped {rows_before - len(df)} rows with missing TotalCharges")

# customerID is just a unique identifier, no predictive value
df = df.drop('customerID', axis=1)

# convert target to 0/1
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print(f"Shape after cleaning: {df.shape}")


# ============================
# 3. Preprocessing
# ============================

print("\nSTEP 3: Preprocessing...")

target = 'Churn'
numeric_features = df.select_dtypes(include=[np.number]).columns.drop(target).tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# One-hot encoding for categorical variables
# using drop_first=True to avoid dummy variable trap
# chose one-hot over label encoding because these categories don't have a natural order
# (like, "DSL" isn't less than "Fiber optic" - they're just different)
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
print(f"Shape after encoding: {df_encoded.shape}")

# split into train and test
X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y  # keeps same churn ratio in both sets
)

print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
print(f"Train churn rate: {y_train.mean()*100:.1f}%, Test churn rate: {y_test.mean()*100:.1f}%")

# scale features - important for logistic regression since it uses gradient descent
# need to fit scaler on train only, then transform both (to avoid data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # just transform, don't fit again!

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


# ============================
# 4. Model Training
# ============================

# --- Logistic Regression (baseline) ---
# starting with this because it's simple and interpretable
# if a simple model works well enough, no need for a complex one

print("\nSTEP 4: Training models...")

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  # handles imbalanced classes
)
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)
print("Logistic Regression - done")

# --- Random Forest ---
# trying this as a comparison - it can capture non-linear patterns
# that logistic regression might miss

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,         # limit depth to prevent overfitting
    min_samples_split=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
print("Random Forest - done")


# ============================
# 5. Evaluation
# ============================

# Not just using accuracy because the dataset is imbalanced
# A model that always predicts "no churn" would get ~73% accuracy
# but would be completely useless - it catches zero churners
# So I'm also looking at precision, recall, and F1

print("\nSTEP 5: Evaluation")
print("=" * 55)

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    Predicted:    No    Yes")
    print(f"    Actual No:  {cm[0][0]:>4}   {cm[0][1]:>4}")
    print(f"    Actual Yes: {cm[1][0]:>4}   {cm[1][1]:>4}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'cm': cm}

lr_metrics = evaluate("Logistic Regression", y_test, lr_preds)
rf_metrics = evaluate("Random Forest", y_test, rf_preds)

# for reference - a dumb model that always says "no churn" would get this accuracy:
print(f"\n(Baseline: always predicting 'No Churn' = {(y_test == 0).mean()*100:.1f}% accuracy)")

# confusion matrix plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (name, metrics) in enumerate([("Logistic Regression", lr_metrics), 
                                         ("Random Forest", rf_metrics)]):
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Churn', 'Churned'],
                yticklabels=['No Churn', 'Churned'],
                annot_kws={'size': 14, 'fontweight': 'bold'})
    axes[idx].set_title(f'{name}\nConfusion Matrix', fontweight='bold', fontsize=12)
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')

plt.tight_layout()
plt.savefig('viz_confusion_matrices.png', bbox_inches='tight')
plt.close()
print("\nSaved: viz_confusion_matrices.png")

# side by side comparison chart
fig, ax = plt.subplots(figsize=(10, 5))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
lr_vals = [lr_metrics['accuracy'], lr_metrics['precision'], lr_metrics['recall'], lr_metrics['f1']]
rf_vals = [rf_metrics['accuracy'], rf_metrics['precision'], rf_metrics['recall'], rf_metrics['f1']]

x = np.arange(len(metrics_names))
width = 0.3

bars1 = ax.bar(x - width/2, lr_vals, width, label='Logistic Regression', 
               color='#3498db', edgecolor='white', alpha=0.85)
bars2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest', 
               color='#e74c3c', edgecolor='white', alpha=0.85)

ax.set_ylabel('Score')
ax.set_title('Model Comparison: All Metrics', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim(0, 1.1)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('viz_model_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: viz_model_comparison.png")


# ============================
# 6. Feature Importance
# ============================

# this is honestly the most useful part for a business
# tells you which factors actually drive customers to leave

print("\nSTEP 6: Feature Importance (from Random Forest)")
print("-" * 45)

feat_imp = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 features:")
for _, row in feat_imp.head(10).iterrows():
    bar = "#" * int(row['Importance'] * 100)
    print(f"  {row['Feature']:<35} {row['Importance']:.4f} {bar}")

# plot top 15
fig, ax = plt.subplots(figsize=(10, 8))
top15 = feat_imp.head(15).sort_values('Importance', ascending=True)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top15)))
ax.barh(top15['Feature'], top15['Importance'], color=colors, edgecolor='white', height=0.6)
ax.set_title('Top 15 Feature Importances (Random Forest)', fontweight='bold', fontsize=13)
ax.set_xlabel('Importance Score')
for i, v in enumerate(top15['Importance']):
    ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('viz_feature_importance.png', bbox_inches='tight')
plt.close()
print("\nSaved: viz_feature_importance.png")

# Interpretation of top features:
# 1. Contract type (two year/one year) - customers with longer contracts churn way less.
#    Makes sense - there's a commitment. Business should push contract upgrades.
# 2. TotalCharges / tenure - these are related. New customers with low total spend
#    are the highest risk. Need better onboarding and early retention offers.
# 3. MonthlyCharges - higher bills = more churn. Customers paying a lot feel they're
#    not getting enough value. Could offer bundled discounts.

print("\nTop 3 features interpretation:")
top3 = feat_imp.head(3)
for _, row in top3.iterrows():
    name = row['Feature']
    if 'Contract' in name:
        print(f"  {name}: Long-term contracts reduce churn significantly.")
        print(f"    -> Suggest: incentivize month-to-month customers to upgrade")
    elif 'TotalCharges' in name:
        print(f"  {name}: Low total spend = new customers = higher churn risk.")
        print(f"    -> Suggest: better onboarding program, early retention offers")
    elif 'tenure' in name:
        print(f"  {name}: New customers leave the most, loyal ones stay.")
        print(f"    -> Suggest: focus retention efforts on first 6-12 months")
    elif 'MonthlyCharges' in name:
        print(f"  {name}: Higher bills correlate with more churn.")
        print(f"    -> Suggest: review pricing, offer bundled deals for value")
    else:
        print(f"  {name}: Importance = {row['Importance']:.4f}")


# ============================
# 7. Final Model Choice
# ============================

print("\n" + "=" * 55)
print("FINAL COMPARISON")
print("=" * 55)
print(f"{'Metric':<12} {'Logistic Reg':>14} {'Random Forest':>14}")
print("-" * 42)
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    lr_val = lr_metrics[metric]
    rf_val = rf_metrics[metric]
    winner = " <--" if lr_val >= rf_val else ""
    winner_rf = " <--" if rf_val > lr_val else ""
    print(f"{metric:<12} {lr_val:>14.4f}{winner} {rf_val:>13.4f}{winner_rf}")

# Picking the model - using F1 as the main metric since it balances precision and recall.
# Also care about recall specifically because missing a churner (false negative)
# is more expensive than a false alarm.
if lr_metrics['f1'] >= rf_metrics['f1']:
    chosen = "Logistic Regression"
    reason = "Better F1 score, plus it's simpler and more interpretable"
else:
    chosen = "Random Forest"
    reason = "Better F1 score, captures non-linear patterns"

print(f"\nChosen model: {chosen}")
print(f"Reason: {reason}")

# save results for reference
with open('results_summary.txt', 'w') as f:
    f.write(f"Logistic Regression:\n")
    f.write(f"  Accuracy:  {lr_metrics['accuracy']:.4f}\n")
    f.write(f"  Precision: {lr_metrics['precision']:.4f}\n")
    f.write(f"  Recall:    {lr_metrics['recall']:.4f}\n")
    f.write(f"  F1-Score:  {lr_metrics['f1']:.4f}\n")
    f.write(f"\nRandom Forest:\n")
    f.write(f"  Accuracy:  {rf_metrics['accuracy']:.4f}\n")
    f.write(f"  Precision: {rf_metrics['precision']:.4f}\n")
    f.write(f"  Recall:    {rf_metrics['recall']:.4f}\n")
    f.write(f"  F1-Score:  {rf_metrics['f1']:.4f}\n")
    f.write(f"\nChosen Model: {chosen}\n")

print(f"\nResults saved to results_summary.txt")
print("\nAll visualizations saved:")
print("  viz_churn_by_contract.png")
print("  viz_churn_by_tenure.png")
print("  viz_charges_distribution.png")
print("  viz_correlation_heatmap.png")
print("  viz_confusion_matrices.png")
print("  viz_model_comparison.png")
print("  viz_feature_importance.png")
print("\nDone!")
