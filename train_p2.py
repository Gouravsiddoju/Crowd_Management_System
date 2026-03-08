from ultralytics import YOLO
import os
import yaml

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_DIR = '/media/gouravsiddoju/Data/PROJECTS/Head_Counter'
DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets', 'crowdhuman_yolo')
DATA_YAML = os.path.join(DATASET_DIR, 'data.yaml')

# P2 CONFIG (Architecture for Tiny Object Detection)
# Based on standard YOLOv8-P2 architecture which adds a high-res P2 layer.
P2_YAML_CONTENT = """
# Ultralytics YOLOv8-P2 🚀, AGPL-3.0 license
# P2-head architecture for tiny object detection
# Parameters
nc: 2  # number of classes (CrowdHuman is usually Head/Person, or just Person)
scales: # model compound scaling constants, i.e. 'n', 's', 'm', 'l', 'x'
  n: [0.33, 0.25, 1024]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, true]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, true]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, true]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, true]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small)

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 2], 1, Concat, [1]]  # cat backbone P2
  - [-1, 3, C2f, [128]]  # 18 (P2/4-xsmall)

  - [[18, 15, 12, 9], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5)
"""

def create_p2_config():
    """Creates the yolov8-p2.yaml config file if it doesn't exist"""
    config_path = os.path.join(PROJECT_DIR, 'yolov8n-p2.yaml')
    if not os.path.exists(config_path):
        print(f"Creating P2 Architecture Config: {config_path}")
        with open(config_path, 'w') as f:
            f.write(P2_YAML_CONTENT)
    return config_path

def train():
    print(f"Checking dataset at: {DATA_YAML}")
    if not os.path.exists(DATA_YAML):
        print("ERROR: Dataset not found!")
        print(f"Please download the dataset and ensure '{DATA_YAML}' exists.")
        print("See DATASET_INSTRUCTIONS.md for details.")
        return

    # 1. Create Model Config
    model_cfg = create_p2_config()
    
    # 2. Initialize Model
    # Logic: If we uploaded 'resumed_last.pt', use it. Otherwise start fresh.
    resume_weights = os.path.join(PROJECT_DIR, 'weights', 'resumed_last.pt')
    
    if os.path.exists(resume_weights):
        print(f"🚀 Found Resume Weights: {resume_weights}")
        print("Loading existing model state to continue training...")
        model = YOLO(resume_weights)
    else:
        print("⚠️ No resume weights found at 'weights/resumed_last.pt'")
        print("Starting FRESH training with YOLOv8n-P2 architecture...")
        # Load COCO weights to backbone, but use P2 config
        model = YOLO(model_cfg).load('yolov8n.pt') 

    print("Starting Training (YOLOv8-P2 on CrowdHuman)...")
    
    # 3. Train
    # CONFIG FLAGGED FOR RTX 4090
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=960,   # High resolution for small heads
        batch=16,    # Reduced to 16 (Safe for 4090 w/ large imgsz)
        workers=16,  # INCREASED for faster data loading
        patience=10,
        device=0,
        project='runs/train',
        name='yolov8n-p2-crowdhuman-4090',
        exist_ok=True,
        # Augmentations for Crowd
        mosaic=1.0,
        mixup=0.1,
    )
    
    print("Training Complete.")
    print(f"Best weights: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train()
