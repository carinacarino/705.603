# main.py
from data_pipeline import Pipeline
from model import Object_Detection_Model, LicensePlate_to_String
import os



if __name__ == "__main__":
    stream_url = 'udp://127.0.0.1:23000'
    output_dir = "./extracted_frames"
    transformed_output_dir = "./transformed_frames"
    cropped_license_plates_dir = "./cropped_license_plates"
    license_plates_strings_dir = "./analysis/results/license_plates.csv"
    lpr_model_path = ('./yolo_lpr/lpr-yolov3.weights', './yolo_lpr/lpr-yolov3.cfg')

    fps = 60  # Desired frame rate for the extracted frames
    duration = 60  # Maximum duration (in seconds) for the frame extraction process


    # Ensure the output directories exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(transformed_output_dir):
        os.makedirs(transformed_output_dir)

    print("Starting frame extraction...")
    Pipeline.extract_frames(stream_url, output_dir, fps, duration)
    print("Frame extraction completed.")

    # Initialize Pipeline with both directories
    pipeline = Pipeline(frame_rate=int(fps), output_dir=output_dir, transformed_output_dir=transformed_output_dir)
    print("About to transform frames...")

    # Transform and Load (Save) the images
    transformed_images = pipeline.transform()
    pipeline.load(transformed_images)

    print("Cropping license plates...")
    Object_Detection_Model.process_directory(transformed_output_dir, cropped_license_plates_dir, lpr_model_path)
    print("Cropping license plates completed.")

    print("Extracting License Plates Characters...")
    lp_to_string = LicensePlate_to_String()
    lp_to_string.extract_license_plates_to_csv(cropped_license_plates_dir, license_plates_strings_dir)





