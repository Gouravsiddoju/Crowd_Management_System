# Robust Dataset Setup (Official CrowdHuman)

Since pre-packaged datasets are failing, we will use the **Official Source** from HuggingFace and convert it ourselves. This guarantees you get the data, and we can specifically target **Heads**.

## 1. Create Structure
Create a folder named `datasets/crowdhuman_raw` inside the project:
```bash
mkdir -p /media/gouravsiddoju/Data/PROJECTS/Head_Counter/datasets/crowdhuman_raw/Images
```

## 2. Download Official Files (HuggingFace)
Download these files and place them in `datasets/crowdhuman_raw/`.
**No Login Required.**

1.  **Annotations (Required)**:
    *   **Option A (Kaggle)**: [leducnhuan/crowdhuman](https://www.kaggle.com/datasets/leducnhuan/crowdhuman). Download the whole thing.
    *   **Option B (HuggingFace)**:
        *   [annotation_train.odgt](https://huggingface.co/datasets/crowdhuman/crowdhuman/resolve/main/annotation_train.odgt)
        *   [annotation_val.odgt](https://huggingface.co/datasets/crowdhuman/crowdhuman/resolve/main/annotation_val.odgt)

2.  **Images**:
    *   If using Kaggle, just extract the images from the zip.
    *   If using HuggingFace, download `CrowdHuman_train01.zip` etc.
    *   [CrowdHuman_train02.zip](https://huggingface.co/datasets/crowdhuman/crowdhuman/resolve/main/CrowdHuman_train02.zip)
    *   [CrowdHuman_train03.zip](https://huggingface.co/datasets/crowdhuman/crowdhuman/resolve/main/CrowdHuman_train03.zip)
    *   [CrowdHuman_val.zip](https://huggingface.co/datasets/crowdhuman/crowdhuman/resolve/main/CrowdHuman_val.zip)

## 3. Extract & Organize
1.  **Unzip** all the image zip files.
2.  **Move** all `.jpg` images into a SINGLE folder: `datasets/crowdhuman_raw/Images/`.
3.  Ensure `annotation_train.odgt` and `annotation_val.odgt` are in `datasets/crowdhuman_raw/`.

Structure should look like:
```
datasets/crowdhuman_raw/
├── annotation_train.odgt
├── annotation_val.odgt
└── Images/
    ├── 273271,c9db9609618e15.jpg
    ├── ... (thousands of images)
```

## 4. Run Conversion
I wrote a script to convert this raw data into YOLO format specifically for **HEAD** detection.
```bash
python3 prepare_crowdhuman.py
```

## 5. Start Training
```bash
python3 train_p2.py
```
