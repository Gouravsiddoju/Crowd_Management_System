"""
Generate annotated video output with bounding boxes for counting verification
"""

import cv2
import torch
from ultralytics import YOLO
import yaml
import argparse
import time

def process_video_with_output(video_path, output_path, model, config, frame_limit=0):
    """Process video and save with bounding boxes - with anti-flicker tracking"""
    
    print(f"\nProcessing: {video_path}")
    print(f"Output: {output_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not create output video")
        return
    
    frame_count = 0
    people_count_history = []
    max_people = 0
    
    frame_skip = config['performance'].get('frame_skip', 1)
    conf_threshold = config['model'].get('conf_threshold', 0.25)
    
    # Cache for detection results to prevent flickering
    cached_detections = []
    cached_count = 0
    
    # Track all unique person IDs seen in the entire video
    unique_person_ids = set()
    
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_limit > 0 and frame_count >= frame_limit:
                print(f"  Reached limit of {frame_limit} frames.")
                break
            
            # Progress indicator
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed
                eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
                print(f"\r  Progress: {frame_count}/{total_frames} ({frame_count*100//total_frames}%) - {fps_processing:.1f} FPS - ETA: {eta/60:.1f}min", end='')
            
            # Process detection on sampled frames
            if frame_count % frame_skip == 0:
                # Run detection with tracking
                results = model.track(frame, verbose=False, conf=conf_threshold, 
                                     device=config['model']['device'], persist=True,
                                     tracker="custom_bytetrack.yaml")
                
                # Clear cache and rebuild
                cached_detections = []
                person_count = 0
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        if int(box.cls[0]) == 0:  # Person class
                            person_count += 1
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            track_id = int(box.id[0]) if box.id is not None else None
                            
                            # Track unique IDs
                            if track_id is not None:
                                unique_person_ids.add(track_id)
                            
                            # Cache detection
                            cached_detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'conf': conf,
                                'track_id': track_id
                            })
                
                cached_count = person_count
                
                people_count_history.append(person_count)
                max_people = max(max_people, person_count)
            
            # Draw cached detections (works for both processed and skipped frames)
            for detection in cached_detections:
                x1, y1, x2, y2 = detection['bbox']
                conf = detection['conf']
                track_id = detection['track_id']
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw tracking ID and confidence
                label = f"ID:{track_id} {conf:.2f}" if track_id is not None else f"{conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw simplified HUD with semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame, (10, 10), (450, 100), (0, 255, 0), 2)
            
            # Display only the two requested metrics
            cv2.putText(frame, f"Total People Detected: {len(unique_person_ids)}", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"People In Frame: {cached_count}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Write frame
            out.write(frame)
    
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    
    finally:
        cap.release()
        out.release()
        
        elapsed = time.time() - start_time
        print(f"\n\n  Finished in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
        print(f"  Max people detected: {max_people}")
        print(f"  Average people: {sum(people_count_history)/len(people_count_history) if people_count_history else 0:.1f}")
        print(f"  Saved to: {output_path}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate annotated video output')
    parser.add_argument('--input', type=str, required=True, help='Input video file')
    parser.add_argument('--output', type=str, required=True, help='Output video file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--model', type=str, help='Override model path')
    parser.add_argument('--limit', type=int, default=0, help='Max frames to process')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load model
    model_name = args.model if args.model else config['model']['name']
    device = config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading {model_name} on {device}...")
    model = YOLO(model_name)
    
    # Process
    process_video_with_output(args.input, args.output, model, config, args.limit)

if __name__ == "__main__":
    main()
