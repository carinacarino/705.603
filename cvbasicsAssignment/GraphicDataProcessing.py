#GraphicDataProcessing.py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


class ObjectDetection:
    """
    This class is used for object detection.
    """
    def __init__(self, net, classes):
        self.net = net
        self.classes = classes
        self.original_image = None

    # A constructor for setting image
    def set_image(self, image_path):
        self.original_image = cv.imread(image_path)
        if self.original_image is None:
            print(f"Failed to load image from {image_path}")

    # Plotting image
    def plot_cv_img(self, input_image):
        plt.imshow(cv.cvtColor(input_image, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    # Detecting objects
    def detect_objects(self, image):
        blob = cv.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers())
        return outs

    # For getting output layers
    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        out_layers = self.net.getUnconnectedOutLayers()

        # Adjusting based on the type of output getUnconnectedOutLayers() returns
        if out_layers.ndim == 2:  # If it's an array of arrays
            output_layers = [layer_names[i[0] - 1] for i in out_layers]
        else:  # If it's a flat array
            output_layers = [layer_names[i - 1] for i in out_layers]

        return output_layers

    # Process detected objects, confidence and class label
    def process_detections(self, outs, width, height, conf_threshold=0.5, nms_threshold=0.4):
        class_ids = []
        confidences = []
        boxes = []

        # First, filter detections by confidence and store relevant info
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression
        indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Filter detections based on NMS and return them along with filtered boxes
        nms_class_ids = [class_ids[i] for i in indices.flatten()]
        nms_confidences = [confidences[i] for i in indices.flatten()]
        nms_boxes = [boxes[i] for i in indices.flatten()]
        return nms_class_ids, nms_confidences, nms_boxes

    # For drawing bounding box around objects
    def draw_bounding_boxes(self, img, class_ids, confidences, boxes):
        # Ensure colors array to match the number of class labels
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        for i, box in enumerate(boxes):
            x, y, w, h = box
            label = str(self.classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, f"{label}: {confidence:.2f}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 2, color, 8)

        return img

    # Adds Gaussian noise
    def add_gaussian_noise(self, image, std):
        row, col, ch = image.shape
        mean = 0
        gauss = np.random.normal(mean, std, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    # Adds salt and pepper noise
    def add_salt_and_pepper_noise(self, image, amount):
        # Add salt-and-pepper noise to the image
        noisy = random_noise(image, mode='s&p', amount=amount)
        noisy_image = np.array(255 * noisy, dtype=np.uint8)
        return noisy_image

    # Resizes images
    def size_experiment(self, scales=np.linspace(0.001, 1.0, 100)):
        if self.original_image is None:
            print("Error: No original image set.")
            return

        results = {}
        for scale in scales:
            resized_image = cv.resize(self.original_image, None, fx=scale, fy=scale)
            outs = self.detect_objects(resized_image)
            class_ids, confidences = self.process_detections(outs, resized_image.shape[1], resized_image.shape[0])
            results[scale] = np.median(confidences) if confidences else 0

        self.plot_results(results, 'Size Scale Factor')

    # Rotates images
    def rotation_experiment(self, angles):
        if self.original_image is None:
            print("Error: No original image set.")
            return

        results = {}
        for angle in angles:
            M = cv.getRotationMatrix2D((self.original_image.shape[1] / 2, self.original_image.shape[0] / 2), angle, 1)
            rotated_image = cv.warpAffine(self.original_image, M,
                                          (self.original_image.shape[1], self.original_image.shape[0]))
            outs = self.detect_objects(rotated_image)
            class_ids, confidences = self.process_detections(outs, rotated_image.shape[1], rotated_image.shape[0])
            results[angle] = np.mean(confidences) if confidences else 0

        self.plot_results(results, 'Rotation Angle')

    # Adds a set of Gaussian or S&P noise to an image
    def noise_experiment(self, noise_type, std_scales):
        if self.original_image is None:
            print("Error: No original image set.")
            return
        original_std = np.std(self.original_image)

        results = {}
        for std_scale in std_scales:
            if noise_type == 'gaussian':
                noisy_image = self.add_gaussian_noise(self.original_image, std_scale*original_std)
                x_title = f"Standard Deviation of Gaussian Noise"
            elif noise_type == 'sp':
                noisy_image = self.add_salt_and_pepper_noise(self.original_image, std_scale)
                x_title = f"Salt & Pepper Noise Amount"
            else:
                print("Unsupported noise type.")
                return

            outs = self.detect_objects(noisy_image)
            class_ids, confidences = self.process_detections(outs, noisy_image.shape[1], noisy_image.shape[0])
            results[std_scale] = np.median(confidences) if confidences else 0

        self.plot_results(results, x_title)

    # Plotting
    def plot_results(self, results, x_title):
        plt.figure(figsize=(10, 6))
        plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-')
        plt.title(f'{x_title} vs. Detection Confidence')
        plt.xlabel(x_title)
        if x_title == 'Standard Deviation of Gaussian Noise':
            std_scales = np.linspace(1, 6, 6)
            std_labels = [f'{scale:.0f}σ' for scale in std_scales]
            plt.xticks(std_scales, labels=std_labels)
        plt.ylabel('Median Detection Confidence')
        plt.grid(True)
        plt.show()


