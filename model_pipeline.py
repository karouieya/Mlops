
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare_data(data_path):
    """
    Charge les données et prépare les variables pour le modèle.
    """

    df = pd.read_csv(data_path)

    # Vérifier les données
    print("Dimensions du dataset :", df.shape)
    print("Valeurs manquantes :")
    print(df.isnull().sum())

    # Encodage de Gender
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Suppression des colonnes inutiles
    df = df.drop(columns=["Surname", "Geography"])

    # Séparation des variables X et de la cible y
    X = df.drop(columns=["Exited", "RowNumber", "CustomerId"])
    y = df["Exited"]

    # Séparation des données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1
    )

    print("Préparation des données terminée.")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Entraîne le modèle Random Forest.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Entraînement terminé.")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n===== ÉVALUATION DU MODÈLE =====")
    print("Accuracy :", accuracy * 100, "%")

    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))

    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    return accuracy


def save_model(model, filename="classifier.joblib"):
    """
    Sauvegarde le modèle entraîné.
    """

    joblib.dump(model, filename)

    print(f"Modèle sauvegardé dans : {filename}")


def load_model(filename="classifier.joblib"):
    """
    Charge un modèle déjà sauvegardé.
    """

    model = joblib.load(filename)

    print("Modèle chargé avec succès.")

    return model
