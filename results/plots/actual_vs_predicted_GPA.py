import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual GPA")
    plt.ylabel("Predicted GPA")
    plt.title("Actual vs Predicted GPA")
    
    plt.savefig("results/plots/outputs/actual_vs_predicted.png", dpi=300)
    plt.show()
    plt.close()