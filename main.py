
import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model
)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Machine Learning de prédiction du churn"
    )

    parser.add_argument(
        "--data",
        default="Churn_Modelling.csv",
        help="Chemin vers le dataset CSV"
    )

    parser.add_argument(
        "--model",
        default="classifier.joblib",
        help="Chemin de sauvegarde du modèle"
    )

    args = parser.parse_args()

    # 1. Préparation des données
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    # 2. Entraînement
    model = train_model(X_train, y_train)

    # 3. Évaluation
    evaluate_model(model, X_test, y_test)

    # 4. Sauvegarde
    save_model(model, args.model)

    # 5. Vérification : chargement du modèle
    loaded_model = load_model(args.model)

    # 6. Test de prédiction
    prediction = loaded_model.predict(X_test.iloc[[0]])

    print("\nPrédiction du premier client du test :", prediction[0])


if __name__ == "__main__":
    main()
