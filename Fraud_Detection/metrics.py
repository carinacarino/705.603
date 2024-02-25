# metrics.py
from sklearn.metrics import precision_score, recall_score
import os


class Metrics:
    """
    Designed for evaluating a fraud detection model.
    """
    def __init__(self):
        # Ensure the results directory exists
        os.makedirs('results', exist_ok=True)

    def precision(self, y_prediction, y_label):
        """Calculate precision."""
        return precision_score(y_label, y_prediction)

    def recall(self, y_prediction, y_label):
        """Calculate recall."""
        return recall_score(y_label, y_prediction)

    def generate_report(self, y_prediction, y_label):
        """Generate a report with precision, recall, sensitivity, and specificity."""
        precision = self.precision(y_prediction, y_label)
        recall = self.recall(y_prediction, y_label)

        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")

        # Generate report text
        report_text = (
            f"Model Results:\n\n"
            f"Precision: {precision * 100:.2f}%\n"
            f"Recall: {recall * 100:.2f}%\n"

        )

        # Write report to a file in the 'results' directory
        with open('results/model_evaluation_report.txt', 'w') as file:
            file.write(report_text)

        print("Report generated in 'results/model_evaluation_report.txt'")
