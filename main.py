import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from typing import Tuple
from zenml import pipeline, step

@step
def load_data() -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    """
    df = pd.read_csv('./flight_data.csv')
    return df

@step
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data.
    """
    # Drop unnecessary columns
    df = df.drop(['time_hour', 'tailnum', 'year', 'sched_dep_time', 'sched_arr_time'], axis=1)

    # Impute missing values for numeric columns
    numeric_columns = ['dep_delay', 'arr_time', 'dep_time', 'air_time', 'arr_delay']
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Convert delay columns to integer
    df['dep_delay'] = df['dep_delay'].astype(int)
    df['arr_delay'] = df['arr_delay'].astype(int)

    # Encode categorical variables
    cat_cols = ['carrier', 'origin', 'dest']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder() 
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Convert dep_delay to binary target variable
    df['Target'] = np.where(df['dep_delay'] > 0, 1, 0)

    return df

@step
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.
    """
    X = df[['carrier', 'origin', 'dest', 'distance', 'hour', 'day', 'month']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

@step
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a random forest classifier model.
    Other models analysis in the notebooks directory
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

@step
def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float]:
    """
    Evaluate the model and return accuracy, MSE, and F1 score.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, mse, f1

@pipeline
def training_pipeline() -> Tuple[float, float, float]:
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    accuracy, mse, f1 = evaluate_model(model, X_test, y_test)
    return accuracy, mse, f1

if __name__ == '__main__':

    '''
    In zenml dashboard, the following box metadata represents

    output0:accuracy_score, output1:mse, output2:f1_score
    '''
    training_pipeline()
