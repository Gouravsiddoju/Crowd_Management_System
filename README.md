<h1 align="center">Crowd Management System</h1>
<p align="center">
  <strong>An Intelligent Multi-Camera Tracking & Vision Enhancement Pipeline for High-Density Environments</strong>
</p>

## 📖 Overview

The **Crowd Management System** is a robust, end-to-end computer vision framework designed for real-time person tracking and occupancy estimation in highly crowded areas (e.g., railway stations, stadiums, and public squares). 

Unlike standard detection pipelines, this system integrates a state-of-the-art **Deep Learning Vision Enhancement Engine (v2.5)** capable of operating in low-light, high-noise environments natively. It intelligently stabilizes, denoises, and exposes raw camera feeds *before* feeding them into a custom YOLO detection network—saving identities that would otherwise be lost in the shadows.

---

## 🌟 Key Features

* **Advanced Vision Enhancement (v2.5)**: GPU/CPU-accelerated pre-processing pipeline including:
  * **Temporal Stabilization**: Reduces heavy camera shake & flicker.
  * **HDRNet Calibrated Exposure**: Brightens ultra-dark areas without blowing out highlights.
  * **FastDVDnet Denoising**: Removes ISO grain natively.
  * **ContourNet Sharpening**: Enhances edge detail on blurry pedestrian contours.
* **Custom YOLO Detection**: Optimized `best.pt` head/person tracker designed specifically for dense crowds.
* **Advanced Re-Identification (ReID)**: Utilizes `DeepFace` embeddings and HSV histograms via Union-Find clustering to match and track identities across multiple camera angles.
* **Polygon-based Spatial Occupancy**: Hardcode inclusion zones and exclusion zones per-camera to accurately compute effective physical area vs capacity.
* **Flow & Analytics Engine**: Live metrics on flow rates, average dwell times, and real-time GPU performance exported automatically to JSON/CSV for web dashboard consumption.
* **Multi-Camera Sync**: Concurrently run unlimited `rtsp` or video file feeds and generate one localized global tracker.

---

## 🏗️ System Architecture

1. **Input Layer**: `StreamHandler` captures `.mp4`, `.wmv`, or `rtsp://` streams utilizing localized queuing buffers to prevent timeouts.
2. **Preprocessing Layer**: Frames are fed into the v2.5 Enhancer or Legacy OpenCV pipeline depending on your configuration.
3. **Detection Core**: YOLO draws high-confidence bounding boxes, feeding them into `BoT-SORT` for Kalman filtering and ID assignment.
4. **Spatial Math Layer**: `ZoneCounter` identifies if a tracked human is inside an actionable zone and updates the capacity.
5. **Multi-Camera Sync**: `CrossCameraMatcher` de-duplicates identical humans seen across different cameras.
6. **Analytics Engine**: Results are logged, and overlaid video streams are broadcast via a REST/Flask server for real-time frontend viewing.

---

## ⚙️ System Requirements

- **OS**: Windows 10/11 or Ubuntu 20.04+
- **Python**: 3.9+
- **GPU**: NVIDIA RTX GPU highly recommended (CUDA 11.8+ for PyTorch) but fully supports CPU execution dynamically.

---

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/Crowd_Management_System.git
   cd Crowd_Management_System
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The system utilizes `lapx` instead of `lap` for Windows compatibility, and `flask` / `flask-cors` for the web stream).*

3. **Verify CUDA/PyTorch Setup** (Optional but recommended):
   ```bash
   python check_gpu.py
   ```

---

## 🛠️ Configuration (`config.yaml`)

The system is highly modular and controlled entirely via `config.yaml`. Before running, ensure your endpoints and thresholds match your physical environment.

### Important Settings to Review:
* **`performance_modes`**: Choose `FAST`, `BALANCED`, or `ACCURATE` to dynamically set YOLO resolution (`imgsz`) and confidence threshold cutoffs based on the power of your machine.
* **`preprocessing`**: Set `use_v2_5_enhancer: true` to unlock the power of deep-learning denoising in dark visual environments.
* **`boundaries`**: Each camera feed requires its own dictionary entry describing the spatial `polygon` inclusion zone to calculate occupancy.

---

## 🏃‍♂️ Usage & Execution

To run the system on a multi-camera RTSP network or local video file:

```bash
python run_multicam.py --video "path/to/your/video.wmv" --config config.yaml
```

**Common Flags:**
- `--video`: Path to local media. If excluded, the system will use streams defined in `config.yaml`.
- `--limit`: Stop the system after X frames (great for testing). Example: `--limit 750` runs for exactly 30 seconds at 25fps.
- `--config`: The path to the runtime configuration file.

### Checking Output:
Upon execution, a timestamped folder will be created locally in the `/results/` directory containing:
1. `output_<video_name>.mp4`: The visually overlaid bounding box video stream.
2. `analytics_<timestamp>.csv`: Tick-by-tick logs of flow rate, unique humans, peak occupancy, and FPS!
3. `depth_map_<video_name>.png`: A 3D approximation of your camera angle.

Enjoy precise, stabilized tracking even in the worst lighting scenarios!
