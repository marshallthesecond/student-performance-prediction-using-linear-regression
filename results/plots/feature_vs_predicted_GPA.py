import matplotlib.pyplot as plt

def plot_feature_vs_predicted(X, y_pred, feature_name):
    if feature_name not in X.columns:
        raise ValueError(f"{feature_name} not found in feature set")

    plt.figure()
    plt.scatter(X[feature_name], y_pred)
    plt.xlabel(feature_name)
    plt.ylabel("Predicted GPA")
    plt.title(f"{feature_name} vs Predicted GPA")
    
    plt.savefig(f"results/plots/outputs/{feature_name}_vs_predicted_GPA.png", dpi=300)
    plt.show()
    plt.close()
