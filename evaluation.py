import numpy as np
import matplotlib.pyplot as plt


def find_tf_fp(y_predicted, y_actual):
    if y_predicted.shape != y_actual.shape:
        raise ValueError("y_pred and y_actual must have the same length")
    tp = int(np.sum((y_predicted == 1) & (y_actual == 1)))
    fp = int(np.sum((y_predicted == 1) & (y_actual == 0)))
    tn = int(np.sum((y_predicted == 0) & (y_actual == 0)))
    fn = int(np.sum((y_predicted == 0) & (y_actual == 1)))
    return tp, fp, tn, fn


def find_stats(y_predicted, y_actual):
    tp, fp, tn, fn = find_tf_fp(y_predicted, y_actual)
    total = tp + tn + fp + fn
    accuracy = ( tp + tn ) / total
    # covering tp + fn = 0 edge case
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    # covering tp + fp = 0 edge case
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    # covering precision + recall = 0 edge case
    pre_rec = precision + recall
    if (pre_rec) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (pre_rec)
    return accuracy, recall, precision, f1


def brier_score(p_test, y_test):
    if p_test.shape != y_test.shape:
        raise ValueError("P and Y do not have the same shape!")
    return np.mean((p_test - y_test) ** 2)


def find_optimal_threshold(p_test, y_test, cost_fn_to_fp_ratio=10):
    # making a precision and recall curve for different threshold values
    # assuming that most of the 'action' is around a threshold of 0.1:

    thresholds = [
        0.0000, 0.0025, 0.0050, 0.0075, 0.0100,
        0.0125, 0.0150, 0.0175, 0.0200,
        0.0225, 0.0250, 0.0275, 0.0300,
        0.0325, 0.0350, 0.0375, 0.0400,
        0.0425, 0.0450, 0.0475, 0.0500,

        0.0510, 0.0520, 0.0530, 0.0540, 0.0550,
        0.0560, 0.0570, 0.0580, 0.0590, 0.0600,
        0.0610, 0.0620, 0.0630, 0.0640, 0.0650,
        0.0660, 0.0670, 0.0680, 0.0690, 0.0700,
        0.0710, 0.0720, 0.0730, 0.0740, 0.0750,
        0.0760, 0.0770, 0.0780, 0.0790, 0.0800,

        0.0805, 0.0810, 0.0815, 0.0820, 0.0825,
        0.0830, 0.0835, 0.0840, 0.0845,
        0.0850, 0.0855, 0.0860, 0.0865, 0.0870,
        0.0875, 0.0880, 0.0885, 0.0890, 0.0895,
        0.0900, 0.0905, 0.0910, 0.0915, 0.0920,
        0.0925, 0.0930, 0.0935, 0.0940, 0.0945,
        0.0950, 0.0955, 0.0960, 0.0965, 0.0970,
        0.0975, 0.0980, 0.0985, 0.0990, 0.0995,
        0.1000,
        0.1005, 0.1010, 0.1015, 0.1020, 0.1025,
        0.1030, 0.1035, 0.1040, 0.1045,
        0.1050, 0.1055, 0.1060, 0.1065, 0.1070,
        0.1075, 0.1080, 0.1085, 0.1090, 0.1095,
        0.1100, 0.1105, 0.1110, 0.1115, 0.1120,
        0.1125, 0.1130, 0.1135, 0.1140, 0.1145,
        0.1150, 0.1155, 0.1160, 0.1165, 0.1170,
        0.1175, 0.1180, 0.1185, 0.1190, 0.1195,
        0.1200,

        0.1300, 0.1400, 0.1500, 0.1600, 0.1700,
        0.1800, 0.1900, 0.2000, 0.2100, 0.2200,
        0.2300, 0.2400, 0.2500, 0.2600, 0.2700,
        0.2800, 0.2900, 0.3000, 0.3100, 0.3200,
        0.3300, 0.3400, 0.3500, 0.3600, 0.3700,
        0.3800, 0.3900, 0.4000, 0.4100, 0.4200,
        0.4300, 0.4400, 0.4500, 0.4600, 0.4700,
        0.4800, 0.4900, 0.5000,

        0.5500, 0.6000, 0.6500, 0.7000, 0.7500,
        0.8000, 0.8500, 0.9000, 0.9500, 1.0000
    ]

    # defining the hypothetical cost of a lending to a defaulter (false negative) and cost of rejecting a good borrower (false positive)
    # cost_fp = 1000 # $, captures lost profit, customer lifetime value etc
    # cost_fn = 10000 # $, captures cost of loss of principal, recovery cost etc
    # Expected cost for a given threshold, t, is:
    # ExC(t) = number_of_false_positives * cost_fp + number_of_false_negatives * cost_fn
    # The ideal threshold to run the model at will minimise this cost:
    best_t = None
    best_cost = float('inf')
    # making a precision recall and ROC plot and finding best cost in one loop
    fpr_plot = []
    tpr_plot = []
    recall_plot = []
    precision_plot = []
    for threshold in thresholds:
        y_pred = (p_test >= threshold).astype(int)
        # finding optimal threshold to minimise cost
        _, fp, _, fn = find_tf_fp(y_pred, y_test)
        cost = fp + fn * cost_fn_to_fp_ratio
        if cost < best_cost:
            best_cost = cost
            best_t = threshold
        # ROC
        tp, fp, tn, fn = find_tf_fp(y_pred, y_test)
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        fpr_plot.append(fpr)
        tpr_plot.append(tpr)
        # precision recall
        _, recall, precision, _ = find_stats(y_pred, y_test)
        recall_plot.append(recall)
        precision_plot.append(precision)

    # Ensure points are sorted by FPR to find the area under the curve via trapezoidal integration
    order = np.argsort(fpr_plot)
    fpr_sorted = np.array(fpr_plot)[order]
    tpr_sorted = np.array(tpr_plot)[order]
    roc_auc = np.trapezoid(tpr_sorted, fpr_sorted)

    return best_t, best_cost, roc_auc, thresholds, fpr_plot, tpr_plot, recall_plot, precision_plot


def plot_roc_and_pr_curves(thresholds, fpr_plot, tpr_plot, recall_plot, precision_plot):
    # plotting ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_plot, tpr_plot, linewidth=2)

    # plotting the individual points on the chart
    label_thresholds = [0.05, 0.0895, 0.10, 0.15, 0.20, 0.50]
    for t in label_thresholds:
        idx = thresholds.index(t)  # find index in your threshold list
        tpr = tpr_plot[idx]
        fpr = fpr_plot[idx]

        plt.scatter(fpr, tpr)  # mark the point
        plt.annotate(
            f"t={t:.2f}",
            (fpr, tpr),
            textcoords="offset points",
            xytext=(5, -5),
            fontsize=9
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)

    # plotting precision recall
    plt.figure(figsize=(8, 6))
    plt.plot(recall_plot, precision_plot, linewidth=2)
    # plotting the individual points on the chart
    label_thresholds = [0.05, 0.0895, 0.10, 0.15, 0.20, 0.50]
    for t in label_thresholds:
        idx = thresholds.index(t)  # find index in your threshold list
        r = recall_plot[idx]
        p = precision_plot[idx]

        plt.scatter(r, p)  # mark the point
        plt.annotate(
            f"t={t:.2f}",
            (r, p),
            textcoords="offset points",
            xytext=(5, -5),
            fontsize=9
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)

def analyse_threshold_vs_cost_ratio(p_test, y_test, cost_ratios=None, thresholds=None):
    # The cost of a false positive is rejecting a worthy lender - ie missing out on fees, and future customer value.
    # The cost of a false negative is lending to a furture defaulter - ie having to spend money retrieving any leftover capital, and not receiving the principal.
    # Naturally, the cost of FN is almost always larger than FP.
    # The cost ratios below are defined as cost_fn / cost_fp
    # GPT generated ratios below - higher ratio means that avoiding losses matters more to you than capturing upside (more risk averse)

    if cost_ratios is None:
        cost_ratios = [
        2,    # very growth-oriented / aggressive
        3,    # fintech / expansion mode
        5,    # mildly risk-averse
        10,   # typical bank / baseline
        15,   # conservative bank
        25,   # stressed / downturn
        50,   # highly risk-averse / capital constrained
        100   # extreme tail-risk avoidance
        ]

    # thresholds as defined earlier in script
    if thresholds is None:
        thresholds = [
        0.0000, 0.0025, 0.0050, 0.0075, 0.0100,
        0.0125, 0.0150, 0.0175, 0.0200,
        0.0225, 0.0250, 0.0275, 0.0300,
        0.0325, 0.0350, 0.0375, 0.0400,
        0.0425, 0.0450, 0.0475, 0.0500,

        0.0510, 0.0520, 0.0530, 0.0540, 0.0550,
        0.0560, 0.0570, 0.0580, 0.0590, 0.0600,
        0.0610, 0.0620, 0.0630, 0.0640, 0.0650,
        0.0660, 0.0670, 0.0680, 0.0690, 0.0700,
        0.0710, 0.0720, 0.0730, 0.0740, 0.0750,
        0.0760, 0.0770, 0.0780, 0.0790, 0.0800,

        0.0805, 0.0810, 0.0815, 0.0820, 0.0825,
        0.0830, 0.0835, 0.0840, 0.0845,
        0.0850, 0.0855, 0.0860, 0.0865, 0.0870,
        0.0875, 0.0880, 0.0885, 0.0890, 0.0895,
        0.0900, 0.0905, 0.0910, 0.0915, 0.0920,
        0.0925, 0.0930, 0.0935, 0.0940, 0.0945,
        0.0950, 0.0955, 0.0960, 0.0965, 0.0970,
        0.0975, 0.0980, 0.0985, 0.0990, 0.0995,
        0.1000,
        0.1005, 0.1010, 0.1015, 0.1020, 0.1025,
        0.1030, 0.1035, 0.1040, 0.1045,
        0.1050, 0.1055, 0.1060, 0.1065, 0.1070,
        0.1075, 0.1080, 0.1085, 0.1090, 0.1095,
        0.1100, 0.1105, 0.1110, 0.1115, 0.1120,
        0.1125, 0.1130, 0.1135, 0.1140, 0.1145,
        0.1150, 0.1155, 0.1160, 0.1165, 0.1170,
        0.1175, 0.1180, 0.1185, 0.1190, 0.1195,
        0.1200,

        0.1300, 0.1400, 0.1500, 0.1600, 0.1700,
        0.1800, 0.1900, 0.2000, 0.2100, 0.2200,
        0.2300, 0.2400, 0.2500, 0.2600, 0.2700,
        0.2800, 0.2900, 0.3000, 0.3100, 0.3200,
        0.3300, 0.3400, 0.3500, 0.3600, 0.3700,
        0.3800, 0.3900, 0.4000, 0.4100, 0.4200,
        0.4300, 0.4400, 0.4500, 0.4600, 0.4700,
        0.4800, 0.4900, 0.5000,

        0.5500, 0.6000, 0.6500, 0.7000, 0.7500,
        0.8000, 0.8500, 0.9000, 0.9500, 1.0000
        ]

    optimal_thresholds = []
    # transforming to np array
    cost_ratios = np.array(cost_ratios)
    thresholds = np.array(thresholds)
    for ratio in cost_ratios:
        costs = []
        for threshold in thresholds:
            y_pred = (p_test >= threshold).astype(int)
            _, fp, _, fn = find_tf_fp(y_pred, y_test)
            cost = fp + fn * ratio
            costs.append(cost)
        best_idx = np.argmin(costs)
        optimal_thresholds.append(thresholds[best_idx])
    # plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(cost_ratios, optimal_thresholds, marker='o', linewidth=2)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Cost to Lender - False Negative / False Positive Ratio')
    plt.ylabel('Optimal Threshold given Cost Ratio')
    plt.title('Cost Ratios vs Optimal Threshold to save Lender Money')

    # Annotate typical bank ratio
    idx = 3  # cost_ratio = 10 (typical bank)
    plt.scatter(cost_ratios[idx], optimal_thresholds[idx], color='red', s=100, zorder=5)
    plt.annotate('Typical Ratio for a Bank', (cost_ratios[idx], optimal_thresholds[idx]),
                 xytext=(10, 15), textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Label each point with coordinates
    for i, (ratio, threshold) in enumerate(zip(cost_ratios, optimal_thresholds)):
        plt.annotate(f'({ratio:.0f}, {threshold:.3f})',
                     (ratio, threshold),
                     xytext=(5, -15), textcoords='offset points',
                     fontsize=8, alpha=0.7)

    return optimal_thresholds, cost_ratios
