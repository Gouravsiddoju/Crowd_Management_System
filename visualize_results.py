"""
Quick visualization script to show detections on sample frames from each video
"""

import cv2
import torch
from ultralytics import YOLO
import yaml
import os
import glob

def visualize_video_sample(video_path, output_path, model, num_frames=5):
    """Extract and annotate sample frames from a video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly throughout the video
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    annotated_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Run detection
        results = model(frame, verbose=False, conf=0.25)
        
        # Draw bounding boxes
        person_count = 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if int(box.cls[0]) == 0:  # Person class
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame {idx}/{total_frames} - {person_count} people", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        annotated_frames.append(frame)
    
    cap.release()
    
    # Create a grid of sample frames
    if annotated_frames:
        # Resize frames to fit in grid
        h, w = annotated_frames[0].shape[:2]
        max_width = 600
        scale = max_width / w
        new_h = int(h * scale)
        
        resized = [cv2.resize(f, (max_width, new_h)) for f in annotated_frames]
        
        # Stack vertically
        grid = cv2.vconcat(resized)
        
        cv2.imwrite(output_path, grid)
        print(f"  Saved visualization: {output_path}")
    
def main():
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model_name = config['model']['name']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading {model_name} on {device}...")
    model = YOLO(model_name)
    
    # Get videos
    video_paths = glob.glob('clip/*.wmv') + glob.glob('clip/*.mp4')
    video_paths.sort()
    
    print(f"\nGenerating visualizations for {len(video_paths)} videos...\n")
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        output_name = f"visualizations/{os.path.splitext(video_name)[0]}_sample.jpg"
        
        print(f"Processing {video_name}...")
        visualize_video_sample(video_path, output_name, model, num_frames=5)
    
    print(f"\n✓ Done! Check the 'visualizations/' folder for sample frames with bounding boxes.")

if __name__ == "__main__":
    main()
