from results.plots.feature_vs_predicted_GPA import plot_feature_vs_predicted
from results.plots.residuals import plot_residuals
from src.data_preprocessing import load_and_clean_data
from src.train_evaluate_model import train_linear_regression, evaluate, feature_importance
# from src.evaluate_model import 
import matplotlib.pyplot as plt
from results.plots.actual_vs_predicted_GPA import plot_actual_vs_predicted

def main(): 
    # loading and cleaning the data
    df = load_and_clean_data("data/Student_performance_data _.csv")

    # training the model
    model, X_test, y_test, metrics = train_linear_regression(df)

    # evaluating the metrics
    eval_metrics = evaluate(model, X_test, y_test)
    print("\n Evaluation Metrics:")
    for metric, value in eval_metrics.items():
        print(f"{metric.upper()}: {value}")

    # feature importance
    importance = feature_importance(model, X_test.columns)
    print("\n Feature Importance:")
    print(importance)
    
    # PLOTTING
    # y_pred = model.predict(X_test)

    # plot_actual_vs_predicted(y_test, y_pred)
    # plot_feature_vs_predicted(X_test, y_pred, "Absences")
    # plot_feature_vs_predicted(X_test, y_pred, "StudyTimeWeekly")
    # plot_residuals(y_test, y_pred)

if __name__ == "__main__":
    main()


