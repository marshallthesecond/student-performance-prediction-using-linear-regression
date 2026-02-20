import matplotlib.pyplot as plt

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted GPA")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted GPA")
    plt.show()
