# Video Output Generation Guide

## Generate Annotated Video with Bounding Boxes

I've created `generate_video_output.py` to process videos and output them with bounding boxes, track IDs, and people counts.

### Improved Detection Settings
Updated `config.yaml` for **maximum accuracy**:
- **Model**: YOLOv8x (Extra Large - best accuracy)
- **Resolution**: 1280px (high detail)
- **Confidence**: 0.20 (lower = more detections)
- **Frame Skip**: 2 (process every other frame for balance)

### Usage

Process a single video:
```bash
python generate_video_output.py --input clip/1.wmv --output output_1_annotated.mp4
```

Process all videos (one at a time):
```bash
python generate_video_output.py --input clip/1.wmv --output output_1.mp4
python generate_video_output.py --input clip/2.wmv --output output_2.mp4
python generate_video_output.py --input clip/3.wmv --output output_3.mp4
python generate_video_output.py --input clip/crowd.wmv --output output_crowd.mp4
```

### Output Format
Each output video will have:
- ✅ Green bounding boxes around each detected person
- ✅ Confidence scores above each box
- ✅ HUD overlay showing:
  - Current frame number
  - Current people count
  - Maximum people count seen

### Speed vs Accuracy
With the improved settings, expect:
- **Video 1 (8025 frames)**: ~15-20 minutes
- **Video 2 (4470 frames)**: ~8-10 minutes
- **crowd.wmv (7447 frames)**: ~12-15 minutes
- **Video 3 (359k frames)**: ~6-8 hours (very long video with 1000 FPS metadata)

**Recommendation**: Start with video 1 or 2 to verify quality before processing all.
