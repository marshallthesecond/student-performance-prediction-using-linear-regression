import json
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

TEST_SIZE = 0.25
RANDOM_STATE = 42

def calculate_metrics(y_test, y_pred):
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

def split_features_target(df, target_column="GPA"):
    return df.drop(columns=[target_column]), df[target_column]

def train_linear_regression(df, target_column="GPA"):

    X, y = split_features_target(df, target_column)

    # dividing the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

# evaluation functions for the model
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred)

# comparing several models with k-fold cross validation, so the scores
# do not depend on one particular train/test split
# the baseline always predicts the mean GPA - every real model should beat it
def compare_models(df, target_column="GPA", n_splits=5):
    X, y = split_features_target(df, target_column)

    models = {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        # Ridge/Lasso penalize coefficient size, so the features are scaled first
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.01)),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }

    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "rmse": -scores["test_rmse"].mean(),
            "mae": -scores["test_mae"].mean(),
            "r2": scores["test_r2"].mean(),
            "r2_std": scores["test_r2"].std()
        })

    return pd.DataFrame(rows).set_index("model")

# feature importance for linear regression
# raw coefficients are "GPA change per 1 unit of the feature", so a feature with a
# wide range (StudyTimeWeekly, 0-20 hours) looks less important than a 0/1 feature
# multiplying by the feature's standard deviation gives the GPA change for a
# one-standard-deviation increase, which can be compared across features
def feature_importance(model, X_train):
    return pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": model.coef_,
        "Standardized": model.coef_ * X_train.std().values
    }).sort_values(by="Standardized", key=abs, ascending=False)

# saving the results and metadata
def save_metadata(path, features, metrics, cv_results):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump({
            "features": list(features),
            "metrics": metrics,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_comparison": cv_results.to_dict(orient="index")
        }, f, indent=4)
