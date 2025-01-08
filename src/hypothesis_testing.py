from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def test_correlation(sample_1, sample_2, alpha=0.05):
    correlation_coefficient, p_value = pearsonr(sample_1, sample_2)
    result_text = (
        "Il y a une corrélation significative positive"
        if p_value < alpha and correlation_coefficient > 0
        else (
            "Il y a une corrélation significative négative"
            if p_value < alpha and correlation_coefficient < 0
            else "Il n'y a pas de corrélation significative"
        )
    )
    return correlation_coefficient, p_value, result_text

def plot_two_samples(sample_1, sample_2, sample_1_label, sample_2_label, figure_title, alternative_plot="scatter"):
    plt.figure(figsize=(8, 6))
    if alternative_plot == "scatter":
        sns.scatterplot(x=sample_1, y=sample_2)
    else:
        sns.histplot(sample_1, kde=True, label=sample_1_label)
        sns.histplot(sample_2, kde=True, label=sample_2_label, color="orange")
        plt.legend()
    plt.title(figure_title)
    plt.xlabel(sample_1_label)
    plt.ylabel(sample_2_label)
    plt.show()