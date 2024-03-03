from flask import Flask, request
import cv2 as cv
import numpy as np
from GraphicDataProcessing import ObjectDetection

app = Flask(__name__)

# Load class names
classes = []
classFilePath = "coco.names"  # Adjust path to your coco.names file
with open(classFilePath, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model configuration and weights
modelConfiguration = "yolov3.cfg"  # Adjust path to your model config file
modelWeights = "yolov3.weights"  # Adjust path to your model weights file
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Initialize the ObjectDetection instance
ot = ObjectDetection(net=net, classes=classes)


@app.route('/detect', methods=['POST'])
def detection():
    # Ensure an image file was uploaded
    if 'imagefile' not in request.files:
        return "No file part", 400

    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return "No image selected for uploading", 400
    imagefile.seek(0)  # Reset file pointer to the beginning

    # Convert the uploaded file to a format suitable for OpenCV
    image = cv.imdecode(np.frombuffer(imagefile.read(), np.uint8), cv.IMREAD_COLOR)

    # Perform object detection
    outs = ot.detect_objects(image)

    # Process detections
    width, height = image.shape[1], image.shape[0]
    class_ids, confidences, boxes = ot.process_detections(outs, width, height)

    # Format the results as a text string
    results_text = "Detected Objects:\n"
    for i, class_id in enumerate(class_ids):
        class_name = ot.classes[class_id]  # Use instance 'ot'
        confidence = confidences[i]
        results_text += f"{class_name} with confidence {confidence:.2f}\n"

    # Return the formatted text string as the response
    return results_text, 200, {'Content-Type': 'text/plain'}


if __name__ == "__main__":
    flaskPort = 8786
    print('Starting server...')
    app.debug = True
    app.run(host='0.0.0.0', port=flaskPort)
