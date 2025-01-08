from src.preprocessing import (
    load_data, select_numerical, handle_missing_values,
    plot_distributions, analyze_correlations, normalize_data
)
from  src.modeling import train_model, evaluate_model, prepare_data, main as modeling_main
from src.hypothesis_testing import test_correlation, plot_two_samples
if __name__ == "__main__":
    data = load_data("./data/database.csv")
    numerical_data = select_numerical(data)
    cleaned_data = handle_missing_values(numerical_data)
    numerical_cols = cleaned_data.columns.tolist()
    plot_distributions(cleaned_data, numerical_cols)
    analyze_correlations(cleaned_data, "taux_d'insertion")
    normalized_data = normalize_data(cleaned_data, numerical_cols)
    prep_data=prepare_data(normalized_data, "taux_d'insertion")
    train_model= train_model(prep_data[0], prep_data[1])
    evaluate_model(train_model, prep_data[0], prep_data[1])
    main=modeling_main(normalized_data, "taux_d'insertion")

    sample_1 = data["Taux d’insertion"]
    sample_2 = data["Salaire net mensuel médian des emplois à temps plein"]

    correlation_coefficient, p_value, result_text = test_correlation(sample_1, sample_2)

    plot_two_samples(
        sample_1=sample_1,
        sample_2=sample_2,
        sample_1_label="Taux d'insertion",
        sample_2_label="Salaire net mensuel médian",
        figure_title="Taux d'insertion vs Salaire",
        alternative_plot="scatter",
    )

    print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
    print(f"P-value: {p_value}")
    print(result_text)