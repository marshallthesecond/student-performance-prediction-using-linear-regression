import argparse
from pathlib import Path

from results.plots.actual_vs_predicted_GPA import plot_actual_vs_predicted
from results.plots.feature_vs_predicted_GPA import plot_feature_vs_predicted
from results.plots.residuals import plot_residuals
from src.data_preprocessing import load_and_clean_data
from src.train_evaluate_model import train_linear_regression, evaluate, compare_models, feature_importance, save_metadata

# paths are built from this file's location, so the script works from any directory
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Student_performance_data.csv"
METADATA_PATH = PROJECT_ROOT / "results" / "metadata.json"

def main():
    parser = argparse.ArgumentParser(description="Student GPA prediction")
    parser.add_argument("--plots", action="store_true", help="save plots to results/plots/outputs")
    args = parser.parse_args()

    # loading and cleaning the data
    df = load_and_clean_data(DATA_PATH)

    # training the model
    model, X_train, X_test, y_train, y_test = train_linear_regression(df)

    # evaluating the metrics
    metrics = evaluate(model, X_test, y_test)
    print("\n Evaluation Metrics (Linear Regression, test set):")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value}")

    # feature importance
    importance = feature_importance(model, X_train)
    print("\n Feature Importance:")
    print(importance)

    # comparing models with cross validation
    cv_results = compare_models(df)
    print("\n Model Comparison (5-fold cross validation):")
    print(cv_results.round(4))

    save_metadata(METADATA_PATH, X_train.columns, metrics, cv_results)
    print(f"\n Results and metadata saved to {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    # PLOTTING
    if args.plots:
        y_pred = model.predict(X_test)

        plot_actual_vs_predicted(y_test, y_pred)
        plot_feature_vs_predicted(X_test, y_pred, "Absences")
        plot_feature_vs_predicted(X_test, y_pred, "StudyTimeWeekly")
        plot_residuals(y_test, y_pred)
        print(" Plots saved to results/plots/outputs")

if __name__ == "__main__":
    main()
