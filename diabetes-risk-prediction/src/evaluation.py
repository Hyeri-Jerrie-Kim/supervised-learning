import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    roc_auc_score, precision_recall_curve
)

def print_classification_report(model, X_test, y_test):
    """
    Print classification report for the provided model on test data.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
    return y_pred


def plot_conf_matrix(y_test, y_pred):
    """
    Plot the confusion matrix for the test predictions.
    """
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc_curve(y_test, y_proba):
    """
    Plot the ROC curve and display the AUC score.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_value = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()


def plot_precision_recall_curve(y_test, y_proba):
    """
    Plot the Precision-Recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance using the Random Forest model from a pipeline.
    Assumes the model was fitted using a pipeline with a VotingClassifier
    containing a RandomForestClassifier with the key 'random_forest'.
    """
    # Extract the VotingClassifier from the pipeline
    voting_clf = model.named_steps["votingclassifier"]
    # Retrieve the RandomForest model using the key 'random_forest'
    rf_model = voting_clf.named_estimators_["random_forest"]
    importances = rf_model.feature_importances_
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()