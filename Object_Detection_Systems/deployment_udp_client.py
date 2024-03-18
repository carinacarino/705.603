import numpy as np
import cv2
import ffmpeg
import sys

def stream_video(input_url, width, height):
    cv2.namedWindow("Video Stream", cv2.WINDOW_NORMAL)

    process1 = (
        ffmpeg
        .input(input_url)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    while True:
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        cv2.imshow("Video Stream", in_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    process1.wait()
    stderr = process1.stderr.read().decode('utf-8')
    if stderr:
        print("FFmpeg Error:", stderr)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting video stream...")
    in_file = 'udp://127.0.0.1:23000'  # Example UDP input URL
    width = 3840  # Example width
    height = 2160  # Example height
    stream_video(in_file, width, height)
