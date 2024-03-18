import numpy as np
import os


class Metrics:
    def __init__(self):
        self.results_dir = "results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def iou(self, boxA, boxB):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        """
        # Similar implementation as before

    def calculate_ap(self, bounding_boxes_pred, bounding_boxes_true, iou_threshold=0.5):
        """
        Calculate Average Precision (AP) for a single class.

        Parameters:
        - bounding_boxes_pred: List of predicted bounding boxes and their scores [(x_min, y_min, x_max, y_max, score), ...].
        - bounding_boxes_true: List of ground truth bounding boxes [(x_min, y_min, x_max, y_max), ...].
        - iou_threshold: Threshold for IoU to consider a detection a true positive.
        """
        # Sort predictions by scores in descending order
        bounding_boxes_pred.sort(key=lambda x: x[-1], reverse=True)

        tp = np.zeros(len(bounding_boxes_pred))
        fp = np.zeros(len(bounding_boxes_pred))
        used_true_boxes = []

        for idx, pred_box in enumerate(bounding_boxes_pred):
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, true_box in enumerate(bounding_boxes_true):
                if gt_idx in used_true_boxes:
                    continue
                iou_score = self.iou(pred_box[:-1], true_box)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp[idx] = 1
                used_true_boxes.append(best_gt_idx)
            else:
                fp[idx] = 1

        # Compute precision and recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / len(bounding_boxes_true)

        # Calculate AP
        ap = np.trapz(precision, recall)
        return ap

    def generate_report(self, bounding_boxes_pred, bounding_boxes_true):
        """
        Generate a report with IoU and mAP for bounding boxes.
        """
        # Assuming a single class, calculate AP directly
        ap = self.calculate_ap(bounding_boxes_pred, bounding_boxes_true)

        report_path = os.path.join(self.results_dir, "metrics_report.txt")
        with open(report_path, "w") as f:
            f.write("Model Results:\n\n")
            f.write(f"mAP: {ap * 100:.2f}%\n")

        print(f"Metrics report generated: {report_path}")