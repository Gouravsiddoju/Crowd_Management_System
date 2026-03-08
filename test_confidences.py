import cv2
import os
import csv
from ultralytics import YOLO

def run_confidence_tests():
    # Setup paths
    video_path = r"31 January\NR_NDLS_SERVER_2_NR_NDLS_PF16_CPSIDE_PTZ_104_20260102183911000_20260102184011000_High.wmv"
    model_path = "best.pt"
    output_dir = "confidence_tests_output"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "confidence_sweep_results.csv")

    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)

    print(f"Opening video: {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    # Extract exactly ONE frame (e.g., frame 50 for a busy scene)
    frame_idx = 50
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) 
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read frame from video.")
        return

    # Save the raw baseline frame
    frame_name = f"frame_{frame_idx}.jpg"
    cv2.imwrite(os.path.join(output_dir, frame_name), frame)

    # Test values map (20%, 30%, 40%, 50%, 60%)
    test_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
    results_data = []

    print("\n--- Starting Confidence Threshold Sweep ---")
    for thresh in test_thresholds:
        print(f"Testing threshold: {thresh * 100}% ...")
        
        # Run inference
        # imgsz=1024 matches the production config for best.pt accuracy
        results = model(frame, conf=thresh, imgsz=1024, verbose=False)
        result = results[0]
        
        # Count total heads detected at this threshold
        detections = len(result.boxes)
        print(f"  -> Found {detections} people.")
        
        results_data.append({
            "conf_threshold": thresh,
            "people_detected": detections,
            "frame_name": frame_name
        })

        # Render the boxes onto the image and save it so the user can visually compare
        annotated_frame = result.plot()
        out_img_path = os.path.join(output_dir, f"detected_thresh_{int(thresh*100)}_{frame_name}")
        cv2.imwrite(out_img_path, annotated_frame)

    # Write to CSV
    print(f"\nSaving results to CSV: {csv_file}")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["conf_threshold", "people_detected", "frame_name"])
        writer.writeheader()
        writer.writerows(results_data)

    print("Done! Tests completed successfully.")

if __name__ == "__main__":
    run_confidence_tests()
