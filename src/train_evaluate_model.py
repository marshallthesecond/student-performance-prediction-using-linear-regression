import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import os
import pandas as pd

def calculate_metrics(y_test, y_pred):
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

def train_linear_regression(df, target_column="GPA"):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # dividing the data into training and testing sets
    test_size = 0.25
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # making predictions
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)

    # saving the results and metadata
    os.makedirs("results", exist_ok=True)

    with open("results/metadata.json", "w") as f:
        json.dump({
            "features": X.columns.tolist(),
            "metrics": metrics,
            "test_size": test_size,
            "random_state": random_state
        }, f, indent=4)

    print(" Results and metadata saved successfully")

    return model, X_test, y_test, metrics

# evaluation functions for the model
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred)

# feature importance for linear regression
# the larger the coefficient, the more important the feature is
def feature_importance(model, feature_names):
    return pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)
