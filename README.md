# 🎥 Simple Re-ID Dataset Builder (Flask App)

This app lets you upload **videos or ZIPs of images** plus **bounding-box JSON** and generates:
- CSVs with detection metadata
- A combined JSON file
- Cropped person images (JPG) in folders
- A human-readable summary

For ZIPs of images, the app **assumes each image represents 1 second** of the original video.

## Try it online (takes a few minute for it to boot up, be patient):
(Due to some constrains, only put a very small set of images or video to avoid server timeout)
- Link: https://reid-dataset-builder.onrender.com/

## DEMO 
<img width="1247" height="667" alt="Screenshot 2025-11-24 at 4 23 42 PM" src="https://github.com/user-attachments/assets/4d9a49d3-6277-431e-8310-1f43eec5449c" />
<img width="1524" height="774" alt="Screenshot 2025-11-24 at 4 24 05 PM" src="https://github.com/user-attachments/assets/ee15a84f-b0a2-4f8a-a905-aa03cbede6f2" />
<img width="1266" height="798" alt="Screenshot 2025-11-24 at 4 23 49 PM" src="https://github.com/user-attachments/assets/0ced93d2-c6be-480b-bcdb-6f078a4f3212" />

---

## Try it locally

## 1. Requirements

- Python 3.9+ (recommended)
- ffmpeg (optional but good for video support on some systems)
- OS: macOS / Linux / Windows

Python dependencies (from `requirements.txt`):

```txt
Flask==3.0.0
opencv-python-headless==4.8.1.78
numpy==1.24.3
pandas==2.0.3
Werkzeug==3.0.1
gunicorn==21.2.0
```

## 2. Setup & Run Locally
1. **Clone or Download** this project
2. In the project folder, create & activate virtual environment
```bash 
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# or:
# .venv\Scripts\activate       # Windows
```
3. Install dependencies
```bash
pip3 install -r requirement_deploy.txt
```
4. Ensure you have:
   - `app_production.py` in the project root
   - `templates/index.html` in `./templates/`
   - Folder structure should look like:
     ```
      .
      ├── app_production.py
      ├── requirement_deploy.txt
      └── templates
          └── index.html
     ```
5. Run the app:
```
python app_production.py
```

6. Open your browser at:
```
http://localhost:5000
```

## 3. Output Files & Structure
```
output/
├── entrance_detections.csv
├── exit_detections.csv          # only if exit provided
├── combined_data.json
├── entrance_crops/
│   ├── entrance_frame000010_det000000.jpg
│   ├── entrance_frame000010_det000001.jpg
│   └── ...
├── exit_crops/                  # only if exit provided
│   ├── exit_frame000020_det000000.jpg
│   └── ...
└── summary.txt
```

1. `entrance_detections.csv` / `exit_detections.csv`
Columns:
- `detection_id` (int)
- `frame_number` (int or key)
- `timestamp_seconds` (float; for ZIP images: 1 image = 1 second)
- `camera_location` ("entrance" or "exit")
- `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`
- `person_id` (nullable)
- `crop_path` (relative path to JPG crop, e.g. entrance_crops/entrance_frame000010_det000000.jpg)
Example:
```
detection_id,frame_number,timestamp_seconds,camera_location,bbox_x,bbox_y,bbox_width,bbox_height,person_id,crop_path
0,10,10.0,entrance,320,180,80,200,1,entrance_crops/entrance_frame000010_det000000.jpg
```

2. `combined_data.json`
Sample Structure:
```
{
  "entrance": {
    "detections": [
      {
        "detection_id": 0,
        "frame_number": 10,
        "timestamp_seconds": 10.0,
        "camera_location": "entrance",
        "bbox": [320, 180, 80, 200],
        "person_id": 1,
        "crop_path": "entrance_crops/entrance_frame000010_det000000.jpg"
      }
      // ...
    ],
    "video_info": {
      "duration_seconds": 120.5,
      "fps": 30.0,
      "total_detections": 123,
      "total_frames": 361,          // or 0 + source_type="images" for ZIPs
      "source_type": "video"        // "images" for ZIP mode
    }
  },
  "exit": {
    "detections": [ /* same schema as entrance */ ],
    "video_info": { /* same structure */ }
  },
  "summary": {
    "total_entrance_detections": 123,
    "total_exit_detections": 45,
    "processed_at": "2025-11-22T12:34:56.789012"
  }
}
```

3. `Crop Folder`
- `entrance_crops/` contains all entrance crops as 128x256 JPGs:
    - From video: entrance_frame000010_det000000.jpg
    - From ZIP images: entrance_<image_stem>_det000000.jpg
- `exit_crops/` same pattern for exit camera.

4. `summary.txt`
Human-friendly text summary:
```
Re-ID Dataset Processing Summary
==================================================

Processed at: 2025-11-22 12:34:56

Entrance Camera:
  Detections: 123
  Duration: 120.5s
  FPS: 30.00

Exit Camera:
  Detections: 45
  Duration: 118.0s
  FPS: 29.97

```
