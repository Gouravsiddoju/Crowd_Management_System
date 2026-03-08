import os
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

# CONFIGURATION
DATASET_ROOT = 'datasets/archive/CrowdHuman'
OUTPUT_ROOT = 'datasets/crowdhuman_yolo'

# Standard CrowdHuman split
ANNOTION_FILES = {
    'train': 'annotation_train.odgt',
    'val': 'annotation_val.odgt'
}
IMAGE_DIRS = {
    'train': 'Images',
    'val': 'Images_val'
}

def convert_odgt_to_yolo(odgt_path, output_labels_dir, output_images_dir, raw_images_dir):
    """
    Converts ODGT annotation line references to YOLO .txt files.
    Extracts 'hbox' (Head Box) specifically for Head Detection.
    """
    if not os.path.exists(odgt_path):
        print(f"Warning: Annotation file not found: {odgt_path}")
        return

    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    
    print(f"Processing {odgt_path}...")
    
    with open(odgt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        data = json.loads(line)
        file_id = data['ID']
        gtboxes = data['gtboxes']
        
        # Image Filename (CrowdHuman images are usually ID.jpg)
        image_name = f"{file_id}.jpg"
        src_image_path = os.path.join(raw_images_dir, image_name)
        dst_image_path = os.path.join(output_images_dir, image_name)
        
        # Check if we have the image (user might not have downloaded all zips yet)
        if not os.path.exists(src_image_path):
            continue
            
        # Copy image to YOLO structure (or symlink to save space)
        # shutil.copy(src_image_path, dst_image_path) 
        if not os.path.exists(dst_image_path):
            os.symlink(os.path.abspath(src_image_path), dst_image_path)
        
        # Get Image Size (need to read it, or assume? CrowdHuman varies)
        # We need actual size to normalize YOLO coordinates.
        # Efficient way: only read if processed. 
        # Actually, let's use PIL/cv2 to get size.
        import cv2
        img = cv2.imread(src_image_path)
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        
        label_txt_path = os.path.join(output_labels_dir, f"{file_id}.txt")
        with open(label_txt_path, 'w') as out_f:
            for box in gtboxes:
                if 'hbox' in box:
                    # Extract Head Box
                    x, y, w, h = box['hbox']
                    
                    # Convert to YOLO (x_center, y_center, w, h) normalized
                    x_center = (x + w / 2) / w_img
                    y_center = (y + h / 2) / h_img
                    w_norm = w / w_img
                    h_norm = h / h_img
                    
                    # Clamp 0-1
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))
                    
                    # Class 0 = Head
                    out_f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print(f"Finished {odgt_path}")

def main():
    # Setup Directories
    if not os.path.exists(DATASET_ROOT):
        print(f"ERROR: Raw dataset not found at {DATASET_ROOT}")
        print("Please create this folder and put annotation_train.odgt and images inside.")
        return

    for split in ['train', 'val']:
        anno_file = os.path.join(DATASET_ROOT, ANNOTION_FILES[split])
        raw_imgs_dir = os.path.join(DATASET_ROOT, IMAGE_DIRS[split]) # Use split specific image dir
        
        out_lbls = os.path.join(OUTPUT_ROOT, split, 'labels')
        out_imgs = os.path.join(OUTPUT_ROOT, split, 'images')
        
        convert_odgt_to_yolo(anno_file, out_lbls, out_imgs, raw_imgs_dir)
        
    # Create data.yaml
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_ROOT)}
train: train/images
val: val/images
names:
  0: head
"""
    with open(os.path.join(OUTPUT_ROOT, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
        
    print("Conversion Complete. Ready to train.")

if __name__ == "__main__":
    main()
