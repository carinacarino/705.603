# Object_Detection_Systems


## Contents
- `metrics_mAP.py`: Contains functions to calculate the mean Average Precision (mAP).
- `models_nms.py`: Contains functions to perform non-maximum suppression post processing.
- `deployment_udp_client.py`: Contains a script to stream videos via UDP.

## Setting up
These set of  requires NumPy (for mAP and NMS) and ffmpeg (for UDP streaming) to handle array operations. Ensure you have NumPy installed:

```sh
python -m pip install numpy ffmpeg-python
```

