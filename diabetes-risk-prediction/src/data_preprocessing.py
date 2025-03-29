import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Pima Indians Diabetes dataset and rename columns.
    """
    df = pd.read_csv(filepath)
    df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                  'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace implausible zero values with NaN and fill missing values using median imputation.
    """
    cols_with_zero_na = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zero_na] = df[cols_with_zero_na].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """
    Split the data into training and test sets using stratified sampling.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)