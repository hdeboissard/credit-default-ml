# creating a machine learning model to predict defaults in loans
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess_data
from model import sigmoid, descend
from evaluation import find_tf_fp, find_stats, brier_score, find_optimal_threshold, plot_roc_and_pr_curves, analyse_threshold_vs_cost_ratio


# Load and preprocess data
x_train, x_test, y_train, y_test = load_and_preprocess_data('loan_default.csv')

# training data and showing log loss plot
beta, train_losses = descend(x_train, y_train)

# trying out the test data
z_test = x_test @ beta
p_test = sigmoid(z_test)
# we now have probability values ( 0 --> 1 ) for each measurement in the x_test data

# tranforming p_test into y_test ( an array of 0s or 1s )
threshold = 0.0895 # optimal threshold which saves the most amount of money as predicted below
y_pred = (p_test >= threshold).astype(int)

brier = brier_score(p_test, y_test)

tp, fp, tn, fn = find_tf_fp(y_pred, y_test)
accuracy, recall, precision, f1 = find_stats(y_pred, y_test)

print("Confusion Matrix (threshold =", threshold, ")")
print("TP:", tp, "FP:", fp)
print("FN:", fn, "TN:", tn)
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1:        {f1:.4f}")
print(f"Brier:     {brier:.4%}")

# Find optimal threshold and generate ROC/PR data
best_t, best_cost, roc_auc, thresholds, fpr_plot, tpr_plot, recall_plot, precision_plot = find_optimal_threshold(p_test, y_test)

print(f"ROC AUC: {roc_auc:.4f}")
print(f"Optimal Cost: ${best_cost:.0f}")
print(f"Optimal Threshold: {best_t:.4f}")

# Plot ROC and PR curves
plot_roc_and_pr_curves(thresholds, fpr_plot, tpr_plot, recall_plot, precision_plot)

# Analyze threshold vs cost ratio
print("\nCost Ratio Analysis")
optimal_thresholds, cost_ratios = analyse_threshold_vs_cost_ratio(p_test, y_test)
print("Cost Ratios:", cost_ratios)
print("Optimal Thresholds:", optimal_thresholds)

# show all plots at once
plt.show()