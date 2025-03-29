import joblib
from data_preprocessing import load_data, clean_data, split_data
from model_training import define_models, train_voting_classifier, evaluate_model_cv
from evaluation import (
    print_classification_report,
    plot_conf_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
)
from tuning import tune_logistic_regression
from inference import predict_sample


def main():
    # 1. Load and preprocess the data
    df = load_data("../data/raw/pima-indians-diabetes.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # 2. Define models and evaluate using cross-validation
    models = define_models()
    print("Cross-Validated F1 Scores:")
    for model_name, model in models.items():
        mean_score, std_score = evaluate_model_cv(model, X_train, y_train)
        print(f"{model_name:<15} | F1 Score (CV): {mean_score:.4f} ± {std_score:.4f}")

    # 3. Train the voting ensemble
    voting_model = train_voting_classifier(X_train, y_train, models)

    # 4. Evaluate the ensemble on the test set
    y_pred = print_classification_report(voting_model, X_test, y_test)
    plot_conf_matrix(y_test, y_pred)
    y_proba = voting_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)

    # 5. Plot feature importance (using the RandomForest from the ensemble)
    plot_feature_importance(voting_model, X_train.columns)

    # 6. Simulate inference on a sample patient
    sample = X_test.iloc[0:1]
    prediction = predict_sample(voting_model, sample)
    print("Prediction for the sample patient:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
    print("\nSample Patient Data:\n", sample)

    # 7. Hyperparameter tuning for Logistic Regression
    best_estimator, best_params, best_score = tune_logistic_regression(X_train, y_train)
    print("Best Parameters:", best_params)
    print("Best F1 Score:", best_score)

    # Save the best tuned logistic regression model
    joblib.dump(best_estimator, "../models/logreg_best.pkl")


if __name__ == "__main__":
    main()
