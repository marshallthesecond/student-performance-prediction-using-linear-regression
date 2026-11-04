from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

def plot_actual_vs_predicted(y_true, y_pred):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual GPA")
    plt.ylabel("Predicted GPA")
    plt.title("Actual vs Predicted GPA")

    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300)
    plt.close()
