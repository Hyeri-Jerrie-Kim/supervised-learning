from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def tune_logistic_regression(X_train, y_train):
    """
    Perform hyperparameter tuning for Logistic Regression using GridSearchCV.
    
    Returns:
        Best estimator, best parameters, and best score.
    """
    pipeline_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )
    param_grid = {"logisticregression__C": [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(
        pipeline_lr, param_grid, scoring="f1", cv=5, verbose=1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_