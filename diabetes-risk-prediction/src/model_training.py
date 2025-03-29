import warnings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")



def define_models() -> dict:
    """
    Define a dictionary of base models with balanced class weights.
    """
    models = {
        "logreg": LogisticRegression(max_iter=1400, class_weight="balanced"),
        "random_forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "svm": SVC(probability=True, class_weight="balanced", random_state=42),
        "decision_tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=42, class_weight="balanced", random_state=42
        ),
        "extra_trees": ExtraTreesClassifier(random_state=42),
    }
    return models


def evaluate_model_cv(model, X_train, y_train):
    """
    Evaluate a given model using stratified 5-fold cross-validation.
    
    Returns:
        mean F1 score and standard deviation.
    """
    pipeline = make_pipeline(StandardScaler(), model)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    return scores.mean(), scores.std()


def train_voting_classifier(X_train, y_train, models: dict):
    """
    Train a soft voting classifier that aggregates predictions from base models.
    
    Returns:
        Fitted voting classifier pipeline.
    """
    pipeline_voting = make_pipeline(
        StandardScaler(),
        VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting="soft",
        ),
    )
    pipeline_voting.fit(X_train, y_train)
    return pipeline_voting
