import numpy as np
import os
import time


class Metrics:
    def __init__(self):
        self.results_dir = "results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.all_bounding_boxes_pred = []
        self.all_bounding_boxes_true = []
        self.start_time = time.time()  # Record the start time

    def iou(self, boxA, boxB):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters:
        - boxA (list): First bounding box in the format [x_min, y_min, x_max, y_max].
        - boxB (list): Second bounding box in the format [x_min, y_min, x_max, y_max].

        Returns:
        - float: IoU value between the two bounding boxes.
        """
        # Extract coordinates from bounding boxes
        xA_min, yA_min, xA_max, yA_max = boxA
        xB_min, yB_min, xB_max, yB_max = boxB

        # Calculate the coordinates of the intersection rectangle
        x_left = max(xA_min, xB_min)
        y_top = max(yA_min, yB_min)
        x_right = min(xA_max, xB_max)
        y_bottom = min(yA_max, yB_max)

        # Check if there is an intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate the area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate the area of both bounding boxes
        boxA_area = (xA_max - xA_min) * (yA_max - yA_min)
        boxB_area = (xB_max - xB_min) * (yB_max - yB_min)

        # Calculate the intersection over union
        iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

        return iou

    def calculate_ap(self, bounding_boxes_pred, bounding_boxes_true, iou_threshold=0.5):
        """
        Calculate the Average Precision (AP) for the predicted bounding boxes.

        Parameters:
        - bounding_boxes_pred (list): List of predicted bounding boxes in the format [x_min, y_min, x_max, y_max].
        - bounding_boxes_true (list): List of ground truth bounding boxes in the format [x_min, y_min, x_max, y_max].
        - iou_threshold (float): IoU threshold for considering a prediction as correct (default: 0.5).

        Returns:
        - float: Average Precision (AP) score.
        """
        tp = 0  # True positives
        fp = 0  # False positives
        ap = 0.0  # Average precision
        matched_gt_indices = set()  # Initialize an empty set to store matched ground truth indices

        for pred_box in bounding_boxes_pred:
            best_iou = 0.0
            best_gt_idx = -1



            for gt_idx, true_box in enumerate(bounding_boxes_true):


                iou_score = self.iou(pred_box, true_box)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt_idx = gt_idx



            if best_iou >= iou_threshold:
                if best_gt_idx not in matched_gt_indices:
                    tp += 1
                    matched_gt_indices.add(best_gt_idx)
                else:
                    fp += 1
            else:
                fp += 1

            precision = tp / (tp + fp)
            ap += precision

        if len(bounding_boxes_pred) > 0:
            ap /= len(bounding_boxes_pred)

        return ap

    def update(self, predictions, ground_truth):
        """
        Update the metrics with the predicted and ground truth bounding boxes.

        Parameters:
        - predictions (list): List of predicted bounding boxes in the format [x, y, w, h].
        - ground_truth (list): List of ground truth bounding boxes in the COCO format [x_min, y_min, width, height].
        """
        # Convert ground truth bounding boxes from COCO format to [x, y, w, h] format
        gt_boxes = []
        for gt_box in ground_truth:
            x_min, y_min, width, height = gt_box
            gt_boxes.append([x_min, y_min, width, height])

        self.all_bounding_boxes_pred.extend(predictions)
        self.all_bounding_boxes_true.extend(gt_boxes)

    def generate_report(self):
        """
        Generate a report based on all stored predictions and ground truths.
        """
        # Calculate AP using all stored predictions and ground truths
        ap = self.calculate_ap(self.all_bounding_boxes_pred, self.all_bounding_boxes_true)

        # Calculate the runtime
        end_time = time.time()
        runtime = end_time - self.start_time

        report_path = os.path.join(self.results_dir, "metrics_report2.txt")
        with open(report_path, "w") as f:
            f.write("Model Results:\n\n")
            f.write(f"mAP: {ap * 100:.2f}%\n")
            f.write(f"Runtime: {runtime:.2f} seconds\n")

        print(f"Metrics report generated: {report_path}")



