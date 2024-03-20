# model.py
import cv2
import numpy as np
import sys
import csv
import os
import pytesseract
# Add the parent directory to sys.path
sys.path.append(os.path.abspath('../'))

from metrics import Metrics
from data_pipeline import Pipeline

class Object_Detection_Model:
    def __init__(self, lpr_model_path):
        self.lpr_model_path = lpr_model_path
        # Load the LPR network
        self.lpr_net = cv2.dnn.readNet(lpr_model_path[0], lpr_model_path[1])


    def predict(self, image):
        lpr_output_layers = self._get_output_layers(self.lpr_net)
        plate_boxes, _, _ = detect_objects(image, self.lpr_net, lpr_output_layers)

        predictions = []
        if plate_boxes:
            for plate_box in plate_boxes:
                x, y, w, h = plate_box
                x_min = x
                y_min = y
                x_max = x + w
                y_max = y + h
                prediction = [x_min, y_min, x_max, y_max]
                predictions.append(prediction)

        return predictions

    @staticmethod
    def crop_image(image, bounding_box):
        """
        Crops an image based on the provided bounding box coordinates.

        Args:
            image (numpy.ndarray): The input image to be cropped.
            bounding_box (list or tuple): A list or tuple containing the bounding box coordinates in the format [x_min, y_min, x_max, y_max].

        Returns:
            numpy.ndarray: The cropped image, or the original image if the bounding box is invalid.
        """
        height, width = image.shape[:2]
        x_min, y_min, x_max, y_max = bounding_box

        # Check if the bounding box coordinates are within the image dimensions
        if x_min < 0 or y_min < 0 or x_max >= width or y_max >= height:
            print(f"Invalid bounding box: {bounding_box} for image with shape {image.shape}. Returning original image.")
            return image

        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image

    @staticmethod
    def process_directory(directory_path, output_directory, lpr_model_path):
        """
        Processes all images in the specified directory and saves the cropped images in the output directory.

        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Load the LPR network
        lpr_net = cv2.dnn.readNet(lpr_model_path[0], lpr_model_path[1])

        # Create an instance of the Object_Detection_Model
        object_detection_model = Object_Detection_Model(lpr_model_path)
        lpr_output_layers = object_detection_model._get_output_layers(lpr_net)

        # Iterate over all files in the input directory
        for filename in os.listdir(directory_path):
            # Check if the file is an image
            if filename.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                image_path = os.path.join(directory_path, filename)
                image = cv2.imread(image_path)

                # Get the predictions (bounding boxes) for the current image
                plate_boxes, _, _ = detect_objects(image, lpr_net, lpr_output_layers)

                # Crop the image based on the bounding boxes and save the cropped images
                for i, (x, y, w, h) in enumerate(plate_boxes):
                    x_min, y_min = x, y
                    x_max, y_max = x + w, y + h
                    bounding_box = [x_min, y_min, x_max, y_max]
                    cropped_image = Object_Detection_Model.crop_image(image, bounding_box)

                    # Skip saving if the cropped image is the same as the original image
                    if np.array_equal(cropped_image, image):
                        continue

                    output_filename = f'cropped_{os.path.splitext(filename)[0]}_{i}.jpg'
                    output_path = os.path.join(output_directory, output_filename)
                    cv2.imwrite(output_path, cropped_image)

    def _get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def test(self, test_dataset):
        pipeline = Pipeline()
        metrics = Metrics()

        for image_path in test_dataset:
            image = cv2.imread(image_path)
            processed_image = pipeline.transform([image])[0]
            predictions = self.predict(processed_image)
            # Assuming you have a mechanism to compare predictions with ground truths
            metrics.update(predictions, image_path)

        metrics.generate_report()



class LicensePlate_to_String:
    def binarize_lp(self, image):
        img_lp = cv2.resize(image, (333, 75))
        img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
        _, img_binary_lp = cv2.threshold(img_gray_lp, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

        LP_WIDTH = img_binary_lp.shape[0]
        LP_HEIGHT = img_binary_lp.shape[1]

        img_binary_lp[0:5, :] = 255
        img_binary_lp[:, 0:5] = 255
        img_binary_lp[60:75, :] = 255
        img_binary_lp[:, 328:333] = 255

        return img_binary_lp

    def extract_license_plate_text(self, cropped_image_path):
        img = cv2.imread(cropped_image_path)

        if img is None or img.size == 0:
            print(f"Invalid image: {cropped_image_path}")
            return "", 0.0

        try:
            bw = self.binarize_lp(img)
            # Ensure tesseract is accessible from the PATH
            custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(bw, config=custom_config)
            confidence = pytesseract.image_to_data(bw, config=custom_config, output_type=pytesseract.Output.DICT)[
                'conf']
            avg_confidence = max(confidence) if confidence else 0.0
            return text.strip(), avg_confidence
        except Exception as e:
            print(f"Error processing image: {cropped_image_path}. Error: {str(e)}")
            return "", 0.0

    def extract_license_plates_to_csv(self, directory_path, output_csv_path):
        # Get a list of image files in the directory
        image_files = [f for f in os.listdir(directory_path) if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]

        # Create a CSV file and write the header
        with open(output_csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Image File', 'License Plate Text', 'Confidence Score'])

            # Iterate over each image file
            for image_file in image_files:
                image_path = os.path.join(directory_path, image_file)

                print(f"Processing image: {image_file}")

                try:
                    # Extract the license plate text from the image
                    license_plate_text, confidence_score = self.extract_license_plate_text(image_path)
                    csv_writer.writerow([image_file, license_plate_text, confidence_score])
                except Exception as e:
                    print(f"Error processing image {image_file}: {str(e)}")

        print(f"License plate extraction completed. Results saved to {output_csv_path}")

def detect_objects(image, net, output_layers):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.75:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

