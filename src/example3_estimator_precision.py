"""
This file is a followup to example_3_probability_estimation, and exploits
data generated by said program.
The results of applying the probability estimator on 1000 connected pairs and
1000 disconnected pairs are loaded. The program uses the difference between
the estimated probability for a pair to be connected and to be disconnected
as basis to set thresholds between "positive" (detected as connected)
and "negative" (detected as disconnected) pairs.
It also shows examples on how to:
-Calculate the precision, recall, and f beta score of an estimation,
 given a threshold (here 0).
-Create a ROC curve for the estimation using various thresholds.
-Create a precision-recall curve for the estimation using various thresholds.
"""

from __future__ import division
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import cPickle as pickle # For Python 2.7
# import _pickle as pickle # For Python 3

def f_beta_score(precision, recall, beta=1):
    """
    Calculates the f beta score of the estimation, given its precision and
    recall, and a value for beta (default is f1).
    Args:
        -precision: the precision value for the estimation (TP/TP+FP)
            +Type: float
        -recall: the recall (true positive ratio) value (TP/P = TP/TP+FN)
            +Type: float
        -beta: the beta value to be used (default is 1)
            +Type: int
    Returns:
        -f_beta: the corresponding f beta score
            +Type: float
    """
    beta_sq = beta**2
    f_beta = (1+beta_sq)*(precision*recall)/((beta_sq*precision)+recall)
    return f_beta

def plot_ROC_Curve(pt_conn, pt_disc):
    """
    Plots the estimation's ROC Curve by using different thresholds,
    and calculating the true and false positive ratios for each.
    Args:
        -pt_conn: a probability table with pairs known to be connected
            +Type: list[((phen_id, gen_id), prob_conn, prob_disc, prob_diff)]
        -pt_disc: a probability table with pairs known to be disconnected
            +Type: list[((phen_id, gen_id), prob_conn, prob_disc, prob_diff)]
    Returns:
        None. Saves result as .png figure.
    """
    # List of thresholds (all positive values).
    thresholds = sorted(set([x[3] for x in pt_conn + pt_disc]))
    # List of collected true positive and false positive ratios.
    ROC_plot = []

    for th in thresholds:
        # Count new true (connected) positives according to this threshold
        nr_TP = len([x for x in pt_conn if x[3] > th])
        # Now do the same thing with false (disconnected) positives
        nr_FP = len([x for x in pt_disc if x[3] > th])

        # Calculate new TPR (True Positive Ratio)
        TPR = nr_TP/len(pt_conn)
        # Calculate corresponding FPR (False Positive Ratio)
        FPR = nr_FP/len(pt_disc)
        # Append the FPR and the TPR to the ROC curve's plot
        ROC_plot.append((FPR, TPR))

    # Remove duplicates, sort the ROC curve by FPR (which is the x-coordinate)
    ROC_plot = list(set([(0.0, 0.0)] + ROC_plot + [(1.0, 1.0)]))
    ROC_plot.sort(key=lambda x: x[0])

    # Draw the pyplot plot for the ROC curve
    plt.plot([x[0] for x in ROC_plot], [x[1] for x in ROC_plot], "b-")
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Ratio')
    plt.ylabel('True Positive Ratio')
    plt.title('ROC curve')
    plt.savefig('../results/ROC_curve.png')
    plt.clf()

def plot_PR_Curve(pt_conn, pt_disc):
    """
    Plots the estimation's precision-recall Curve by using different thresholds,
    and calculating the precision and recall values for each.
    Args:
        -pt_conn: a probability table with pairs known to be connected
            +Type: list[((phen_id, gen_id), prob_conn, prob_disc, prob_diff)]
        -pt_disc: a probability table with pairs known to be disconnected
            +Type: list[((phen_id, gen_id), prob_conn, prob_disc, prob_diff)]
    Returns:
        None. Saves result as .png figure.
    """
    # List of thresholds (all positive values).
    thresholds = sorted(set([x[3] for x in pt_conn + pt_disc]))
    # List of collected precisions and recalls.
    PR_plot = []

    for th in thresholds:
        # Count new true positives and true negatives according to this threshold
        nr_TP = len([x for x in pt_conn if x[3] >= th])
        nr_FP = len([x for x in pt_disc if x[3] >= th])

        # Calculate new precisions and recalls for connected and disconnected pairs
        precision = nr_TP/(nr_TP+nr_FP)
        recall = nr_TP/len(pt_conn)

        # Append the precisions and recalls to their respective plots
        PR_plot.append((recall, precision))

    # Remove duplicates, sort the PR curves by recall (which is the x-coordinate)
    PR_plot = list(set([(0.0, 1.0)] + PR_plot + [(1.0, 0.0)]))
    PR_plot.sort(key=lambda x: x[0])

    # Draw the pyplot plot for the PR curves
    plt.plot([x[0] for x in PR_plot], [x[1] for x in PR_plot], "b-")
    plt.plot([0, 1], [1, 0], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.savefig('../results/PR_curve.png')
    plt.clf()

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# Get the probability table from pickle files, deduce number of connected and disconnected pairs
prob_table_path_conn = "../results/estimated_probs_conn.pkl"
prob_table_path_disc = "../results/estimated_probs_disc.pkl"

prob_table_conn = pickle.load(open(prob_table_path_conn, 'rb'))
prob_table_disc = pickle.load(open(prob_table_path_disc, 'rb'))
shouldbe_pos_num = len(prob_table_conn)
shouldbe_neg_num = len(prob_table_disc)

# Get true and false positives (connected) and negatives (disconnected)
true_pos = [x for x in prob_table_conn if x[3] > 0]
false_pos = [x for x in prob_table_disc if x[3] > 0]
true_pos_num = len(true_pos)
false_pos_num = len(false_pos)
false_neg_num = shouldbe_pos_num - true_pos_num
true_neg_num = shouldbe_neg_num - false_pos_num

# Relevant statistics: precision, recall, f1-score
precision_conn = true_pos_num/(true_pos_num+false_pos_num)
recall_conn = true_pos_num/shouldbe_pos_num
f1_score_conn = f_beta_score(precision_conn, recall_conn)

precision_disc = true_neg_num/(true_neg_num+false_neg_num)
recall_disc = true_neg_num/shouldbe_neg_num
f1_score_disc = f_beta_score(precision_disc, recall_disc)

# Print calculated values in console
print "---------------------------------------"
print "COLLECTED DATA (CONNECTED)"
print "Phenotype\tGenotype\tPrConn\tPrDisc\tDifference"
random_pos = random.sample(range(true_pos_num), 4)
for i in random_pos:
    print prob_table_conn[i][0][0], "\t", prob_table_conn[i][0][1], "\t", prob_table_conn[i][1], "\t", prob_table_conn[i][2], "\t", prob_table_conn[i][3], "\t"
print "---------------------------------------"
print "COLLECTED DATA (DISCONNECTED)"
print "Phenotype\tGenotype\tPrConn\tPrDisc\tDifference"
random_pos = random.sample(range(false_pos_num), 4)
for i in random_pos:
    print prob_table_disc[i][0][0], "\t", prob_table_disc[i][0][1], "\t", prob_table_disc[i][1], "\t", prob_table_disc[i][2], "\t", prob_table_disc[i][3], "\t"
print "---------------------------------------"
print "---------------------------------------"
print "Total connected (positive) =\t", shouldbe_pos_num
print "Total disconnected (negative) =\t", shouldbe_neg_num
print "---------------------------------------"
print "True positives =\t", true_pos_num
print "False positives =\t", false_pos_num
print "---------------------------------------"
print "Precision (conn) =\t", precision_conn
print "Recall (conn) =\t\t", recall_conn
print "F1-Score (conn) =\t", f1_score_conn
print "---------------------------------------"
print "Precision (disc) =\t", precision_disc
print "Recall (disc) =\t\t", recall_disc
print "F1-Score (disc) =\t", f1_score_disc
print "---------------------------------------"

# Plot the ROC and PR curves, and save them as .png plots
plot_ROC_Curve(prob_table_conn, prob_table_disc)
plot_PR_Curve(prob_table_conn, prob_table_disc)
