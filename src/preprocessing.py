import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from unidecode import unidecode

import pandas as pd
from unidecode import unidecode


def load_data(path: str, delimiter: str = ";", encoding: str = "utf-8", header: int = 0) -> pd.DataFrame:
    # Augmenter le nombre de colonnes affichées dans pandas
    pd.set_option('display.max_columns', None)

    # Charger les données CSV
    data = pd.read_csv(path, delimiter=delimiter, encoding=encoding, header=header)

    # Standardisation des noms de colonnes
    data.columns = [unidecode(col) for col in data.columns]  # enlever les caractères spéciaux
    data.columns = data.columns.str.strip().str.replace(' ', '_')  # supprimer les espaces
    data.columns = data.columns.str.lower()  # Uniformiser les noms et éviter les problèmes liés à la casse

    # Afficher les noms de colonnes après nettoyage
    print("Après nettoyage:", data.columns)

    return data
def select_numerical(data: pd.DataFrame) -> pd.DataFrame:
    return data.select_dtypes(include=["number"])

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    return data.fillna(data.mean())

def plot_distributions(data: pd.DataFrame, numerical_cols: list):
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f"Distribution de {col}")
        plt.show()

def analyze_correlations(data: pd.DataFrame, target_column: str):
    correlations = data.corr()[target_column].sort_values(ascending=False)
    print(f"Corrélations avec {target_column} :\n{correlations}\n")

def normalize_data(data: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
    scaler = StandardScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    if "Salaire net mensuel médian des emplois à temps plein" in data.columns and "Taux d’insertion" in data.columns:
        data["Salaire_par_insertion"] = data["Salaire net mensuel médian des emplois à temps plein"] / data["Taux d’insertion"]
    return data