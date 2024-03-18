# CASE STUDY: AUTOMATED LICENSE PLATE RECOGNITION

___
## RUNNING THE ALPR SYSTEM LOCALLY

1. Open a terminal where your video is located and run a server:
```shell
ffmpeg -i LicensePlateReaderSample_4K.mov -vcodec mpeg4 -f mpegts -t 60 udp://127.0.0.1:23000
```
![img.png](processing-pipeline/img.png)

2. Open another terminal where the main.py script is located and run it:
```python
python main.py
```
Note: ffmpeg.exe will run for a set duration (60 seconds) and will terminate before moving on to transformation.

![img_1.png](processing-pipeline/img_1.png)

![img_2.png](processing-pipeline/img_2.png)

___
## ALPR SYSTEM with DOCKER
![docker_compose.png](processing-pipeline/docker_compose.png)

![img_4.png](processing-pipeline/img_4.png)
The system utilizes a multi-container Docker setup, ensuring modular architecture, where each container serves a unique role in the overall application. Below is a description of each container and instructions on how to run them.

### Step 1: Define Docker Containers

#### Video Streamer Container

The `video-streamer` container is responsible for capturing video streams and possibly preprocessing the video data before sending it to the processing pipeline. This container uses FFmpeg or a similar tool to handle video streaming. 

#### Processing Pipeline Container

The `processing-pipeline` container takes the video input from the `video-streamer` container, extracts frames, identifies license plates within those frames, and uses Optical Character Recognition (OCR) to interpret the license plate numbers. The OCR results, including the recognized text and confidence scores, are saved to a CSV file for review.

### Step 2: Docker Compose

Use Docker Compose to orchestrate the containers.

To run the system, make sure you're in the project directory where the `docker-compose.yml` file is located.
```shell
docker-compose build
```
To start up the services: 
```shell
docker-compose up
```

![img_5.png](processing-pipeline/img_5.png)

![img_6.png](processing-pipeline/img_6.png)
___
## OCR LICENSE PLATE IMAGE TO TEXT RESULTS

### Output Format
![img_3.png](processing-pipeline/img_3.png)
Our system processes a series of images to detect and recognize license plates. The results are saved in a CSV file. The CSV file contains three columns:

1. **Image File**: The name of the image file processed. This allows users to trace back the results to the license plate image to review.

2. **License Plate Text**: The text predicted by the OCR as present on the license plate in the image.

3. **Confidence Score**: A numerical value representing the confidence level of the OCR prediction. The confidence score ranges from 0 to 100, where 0 indicates no confidence and 100 indicates full confidence in the prediction.


## Reviewing Results

The CSV file is designed to assist in the manual verification process. Users can cross-reference the image files with the predicted text and use the confidence score to prioritize which results to review first. Typically, lower confidence scores might require more immediate attention, as they indicate less certainty in the OCR's accuracy.

To facilitate this review, one might follow these steps:

1. Open the CSV file in a spreadsheet program like Microsoft Excel or Google Sheets for a user-friendly way to sort and filter the results.

2. Sort the results by the Confidence Score column to review lower-scoring predictions first, as these are more likely to contain inaccuracies.

3. For each entry, locate the corresponding image file to visually confirm the OCR prediction.

4. Make any necessary corrections directly within the spreadsheet or a separate verification system as required by your workflow.

## CSV File Location

The CSV file is saved in the following location within the project directory:

> ./analysis/results/license_plates.csv

## Purpose

The purpose of saving OCR results in this CSV format is to create an efficient, scalable method for verifying and correcting OCR predictions. It streamlines the review process for large batches of images, ultimately enhancing the accuracy of the data captured by the OCR system.