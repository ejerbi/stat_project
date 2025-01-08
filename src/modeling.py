import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def prepare_data(data: pd.DataFrame, target_column: str):
    """Prépare les données en séparant les features et la cible."""
    X = data.drop(columns=[target_column])  # Séparer les features
    y = data[target_column]  # Variable cible
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Entraîner un modèle de régression linéaire."""
    model = LinearRegression()
    model.fit(X_train, y_train)  # Entraînement du modèle
    return model


def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series):
    """Évaluer le modèle avec les données de test."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def main(data: pd.DataFrame, target_column: str):
    """Processus complet de création, entraînement et évaluation du modèle."""
    X, y = prepare_data(data, target_column)

    # Diviser en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle
    model = train_model(X_train, y_train)

    # Évaluer le modèle
    mse, r2 = evaluate_model(model, X_test, y_test)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    return model