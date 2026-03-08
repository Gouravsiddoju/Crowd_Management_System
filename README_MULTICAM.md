# Multi-Camera Person Counting System

This system processes multiple CCTV videos from a folder (`clip/`), tracks people in each video, and then identifies unique individuals across all videos using Re-Identification (Re-ID).

## Features
- **Batch Processing**: Automatically processes all videos in the `clip` folder.
- **Accurate Tracking**: Uses YOLOv8 + BoT-SORT for robust tracking within each video.
- **Cross-Camera Re-ID**: Matches people across different videos using:
  - **Face Recognition**: Uses DeepFace (FaceNet512) for high-precision matching.
  - **Appearance Matching**: Uses color histograms and feature matching for when faces are not visible.
- **Global Counting**: Outputs the total number of unique people found across the entire station.

## Requirements
Ensure you have the dependencies installed:
```bash
pip install -r requirements.txt
pip install deepface scipy
```

## How to Run
1. Place your videos in the `clip` folder.
2. Run the multi-camera script:
   ```bash
   python run_multicam.py
   ```
   
   Optional arguments:
   - `--folder`: Specify custom video folder (default: `clip`)
   - `--config`: Specify custom config file (default: `config.yaml`)

## Output
The script will print progress for each video and finally a Global Analysis report:
```
FINAL REPORT:
  Total Videos Processed: 4
  Total Unique Humans Across All Cameras: 142
----------------------------------------
  Breakdown by Camera (Local Counts):
    - 1.wmv: 45 people
    - 2.wmv: 60 people
    ...
```

## Performance Note
Processing high-resolution video files (checking every frame for small faces) is computationally intensive.
- Expect it to take time proportional to the video length.
- For faster processing, you can increase `frame_skip` in `config.yaml` under `performance` section, but this might reduce tracking accuracy.
