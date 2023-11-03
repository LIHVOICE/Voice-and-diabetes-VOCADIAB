from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.utils import resample
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import numpy as np

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)

def bootstrap_factors(df, factors, n_iterations=1000, plot=False):
    # Set random seed
    np.random.seed(42)
    
    results = {
        'Key factor': [],
        'Threshold': [],
        'Metric': [],
        'Score': []
    }

    # DataFrame to store test results
    test_results = pd.DataFrame(columns=['Key factor', 'Metric', 'Above Threshold', 'Below Threshold', 'Statistic', 'adjusted P-value'])

    for factor in factors:
        for threshold in [0, 1]:
            values = df[df[factor] == threshold][['diabetes', 'predictions_binary']].values

            # Bootstrapping
            for i in range(n_iterations):
                bootstrapped = resample(values)
                sensitivity_score = sensitivity(bootstrapped[:, 0], bootstrapped[:, 1])
                specificity_score = specificity(bootstrapped[:, 0], bootstrapped[:, 1])

                if len(np.unique(bootstrapped[:, 0])) > 1:
                    auc = roc_auc_score(bootstrapped[:, 0], bootstrapped[:, 1])
                else:
                    auc = np.nan

                balanced_accuracy = balanced_accuracy_score(bootstrapped[:, 0], bootstrapped[:, 1])

                results['Key factor'].extend([factor]*4)
                results['Threshold'].extend(['Above Threshold' if threshold else 'Below Threshold']*4)
                results['Metric'].extend(['ROC-AUC', 'balanced_accuracy', 'specificity', 'sensitivity'])
                results['Score'].extend([auc, balanced_accuracy, specificity_score, sensitivity_score])

    results_df = pd.DataFrame(results)

    # Perform tests and adjust p-values
    p_values = []

    for factor in factors:
        for metric in ['ROC-AUC', 'balanced_accuracy', 'specificity', 'sensitivity']:
            scores_above = results_df[(results_df['Key factor'] == factor) & (results_df['Metric'] == metric) & (results_df['Threshold'] == 'Above Threshold')]['Score']
            scores_below = results_df[(results_df['Key factor'] == factor) & (results_df['Metric'] == metric) & (results_df['Threshold'] == 'Below Threshold')]['Score']

            avg_std_above = f"{scores_above.mean():.2f} ({scores_above.std():.2f})"
            avg_std_below = f"{scores_below.mean():.2f} ({scores_below.std():.2f})"

            statistic, p_value = mannwhitneyu(scores_above, scores_below)
            p_values.append(p_value)

    # Adjust the p-values for multiple testing
    adjusted_pvals = multipletests(p_values, alpha=0.05, method='bonferroni')[1]

    # Add results to the test_results DataFrame
    index = 0
    for factor in factors:
        for metric in ['ROC-AUC', 'balanced_accuracy', 'specificity', 'sensitivity']:
            scores_above = results_df[(results_df['Key factor'] == factor) & (results_df['Metric'] == metric) & (results_df['Threshold'] == 'Above Threshold')]['Score']
            scores_below = results_df[(results_df['Key factor'] == factor) & (results_df['Metric'] == metric) & (results_df['Threshold'] == 'Below Threshold')]['Score']

            avg_std_above = f"{scores_above.mean():.2f} ({scores_above.std():.2f})"
            avg_std_below = f"{scores_below.mean():.2f} ({scores_below.std():.2f})"

            statistic, _ = mannwhitneyu(scores_above, scores_below)
            test_results = test_results.append({
                'Key factor': factor, 
                'Metric': metric, 
                'Above Threshold': avg_std_above, 
                'Below Threshold': avg_std_below, 
                'Statistic': statistic, 
                'adjusted P-value': adjusted_pvals[index]
            }, ignore_index=True)
            index += 1

    # Plot boxplots if required
    if plot:
        metrics = ['ROC-AUC', 'balanced_accuracy', 'specificity', 'sensitivity']
        for metric in metrics:
            plt.figure(figsize=(15, 7))
            sns.boxplot(data=results_df[results_df['Metric'] == metric], x='Key factor', y='Score', hue='Threshold')
            plt.xlabel('Key Factor', fontsize=14)
            plt.ylabel('Score', fontsize=14)
            plt.title(f"Vocal biomarker's {metric} distribution for each key factor", fontdict={'size': 16})
            plt.legend(title='Threshold')
            plt.show()
    return results_df, test_results