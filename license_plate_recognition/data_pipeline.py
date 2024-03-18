# data_pipeline.py
import cv2
import os
import ffmpeg
import numpy as np
import subprocess
import re
import time
class Pipeline:
    def __init__(self, frame_rate, output_dir, transformed_output_dir):
        self.frame_rate = frame_rate
        self.output_dir = output_dir
        self.transformed_output_dir = transformed_output_dir  # Add this line
        # Ensure output directories exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.transformed_output_dir):  # Ensure transformed output directory exists
            os.makedirs(self.transformed_output_dir)

    import ffmpeg
    import time

    @staticmethod
    def extract_frames(stream_url, output_dir, fps, duration=10):
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Construct the output pattern
        output_pattern = os.path.join(output_dir, 'frame-%04d.jpeg')

        # Execute the FFmpeg command
        stream = (
            ffmpeg
            .input(stream_url)
            .filter('fps', fps=fps)
            .output(output_pattern, start_number=0, shortest=None)  # Add shortest=None to disable shortest mode
            .overwrite_output()
        )

        # Run the FFmpeg command asynchronously
        process = stream.run_async(pipe_stdout=True, pipe_stderr=True)

        print('Terminating ffmpeg... ')
        # Wait for the specified duration
        time.sleep(duration)

        # Terminate the FFmpeg process
        process.terminate()

        # Wait for the process to terminate
        process.communicate()

        # Check the exit code
        if process.returncode == 0:
            print("Frame extraction completed.")
        else:
            print(f"Frame extraction failed with exit code {process.returncode}")

    def transform(self):
        crop_region = ((849, 1119), (3063, 2151))
        transformed_images = []  # Initialize an empty list
        print(f"Starting transformation on frames in {self.output_dir}")
        for frame_name in sorted(os.listdir(self.output_dir)):
            frame_path = os.path.join(self.output_dir, frame_name)
            print(f"Attempting to transform frame: {frame_path}")
            if not frame_name.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                continue  # Skip non-image files

            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Failed to read image: {frame_path}")
                continue
            print(f"Processing {frame_name}, shape: {frame.shape}")

            # Apply transformations
            cropped_frame = frame[crop_region[0][1]:crop_region[1][1], crop_region[0][0]:crop_region[1][0]]
            upscaled_frame = cv2.resize(cropped_frame, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            transformed_images.append(upscaled_frame)

        print(f"Transformed {len(transformed_images)} frames.")
        return transformed_images

    def load(self, transformed_images):
        for idx, img in enumerate(transformed_images):
            transformed_frame_path = os.path.join(self.transformed_output_dir, f"transformed_frame_{idx + 1:04d}.jpeg")
            cv2.imwrite(transformed_frame_path, img)
        print(f"Saved {len(transformed_images)} transformed frames")
