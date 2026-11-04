from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted GPA")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted GPA")

    plt.savefig(OUTPUT_DIR / "residuals.png", dpi=300)
    plt.close()
