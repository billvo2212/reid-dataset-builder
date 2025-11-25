"""
Production-ready Flask Web Application for Simple Re-ID Dataset Builder
Optimized to avoid memory explosions on long videos
"""

from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import json
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import re
import time  # <-- for processing duration

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', tempfile.mkdtemp())

# Create upload folder if it doesn't exist
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v'}
ALLOWED_JSON_EXTENSIONS = {'json'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
ALLOWED_ARCHIVE_EXTENSIONS = {'zip'}


def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


class SimpleReIDProcessor:
    """Simple Re-ID processor for web app"""

    def __init__(self, resize=(128, 256)):
        self.resize = resize

    def load_bounding_boxes(self, bboxes_file):
        """Load and normalize bounding boxes"""
        with open(bboxes_file, 'r') as f:
            bboxes_data = json.load(f)

        normalized_bboxes = {}
        for key, value in bboxes_data.items():
            if key.startswith('frame_'):
                frame_num = int(key.split('_')[1])
            else:
                frame_num = int(key)

            frame_bboxes = []
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        bbox = item.get('bbox', item.get('box', None))
                        person_id = item.get('person_id', item.get('id', None))
                        if bbox:
                            frame_bboxes.append({
                                'bbox': bbox,
                                'person_id': person_id
                            })
                    elif isinstance(item, list) and len(item) == 4:
                        frame_bboxes.append({
                            'bbox': item,
                            'person_id': None
                        })

            if frame_bboxes:
                normalized_bboxes[frame_num] = frame_bboxes

        return normalized_bboxes

    def crop_person(self, frame, bbox):
        """Crop and resize person"""
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(frame_w, int(x + w))
        y2 = min(frame_h, int(y + h))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        resized = cv2.resize(crop, self.resize, interpolation=cv2.INTER_LINEAR)
        return resized

    def _prepare_crops_dir(self, crops_dir):
        if crops_dir is None:
            return None
        crops_path = Path(crops_dir)
        crops_path.mkdir(parents=True, exist_ok=True)
        return crops_path

    def process_video(self, video_path, bboxes_file, camera_location, crops_dir=None):
        """Process single video; saves crops as images and returns metadata"""
        bboxes = self.load_bounding_boxes(bboxes_file)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Fallback if FPS is invalid or not present
        if video_fps is None or video_fps <= 0:
            video_fps = 1.0

        duration = total_frames / video_fps

        frames_to_process = sorted(bboxes.keys())

        detections = []
        current_frame = 0
        crops_path = self._prepare_crops_dir(crops_dir)

        for target_frame in frames_to_process:
            if current_frame != target_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                current_frame = target_frame

            ret, frame = cap.read()
            if not ret:
                continue

            timestamp_seconds = target_frame / video_fps if video_fps > 0 else 0

            for bbox_data in bboxes[target_frame]:
                bbox = bbox_data['bbox']
                person_id = bbox_data['person_id']

                crop = self.crop_person(frame, bbox)
                if crop is None:
                    continue

                crop_rel_path = None
                if crops_path is not None:
                    crop_filename = f"{camera_location}_frame{target_frame:06d}_det{len(detections):06d}.jpg"
                    full_path = crops_path / crop_filename
                    # Save in BGR (OpenCV) – training script can read with cv2 or PIL
                    cv2.imwrite(str(full_path), crop)
                    crop_rel_path = f"{crops_path.name}/{crop_filename}"

                detections.append({
                    'detection_id': len(detections),
                    'frame_number': int(target_frame),
                    'timestamp_seconds': float(timestamp_seconds),
                    'camera_location': camera_location,
                    'bbox': bbox,
                    'person_id': person_id,
                    'crop_path': crop_rel_path
                })

            current_frame += 1

        cap.release()

        return {
            'detections': detections,
            'video_info': {
                'duration_seconds': duration,
                'fps': float(video_fps),
                'total_detections': len(detections),
                'total_frames': total_frames
            }
        }

    def process_images(self, images_dir, bboxes_file, camera_location, crops_dir=None):
        """
        Process folder of images with annotations.

        Assumption: each image corresponds to 1 second from the video.
        The timestamp is inferred from the frame key or numeric part of filename.
        """
        bboxes = self.load_bounding_boxes(bboxes_file)

        images_path = Path(images_dir)
        image_files = {}

        # Map frame numbers/names to image files
        for img_file in images_path.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                stem = img_file.stem
                try:
                    match = re.search(r'(\d+)', stem)
                    if match:
                        frame_num = int(match.group(1))
                    else:
                        frame_num = stem
                    image_files[frame_num] = img_file
                except Exception:
                    image_files[stem] = img_file

        detections = []
        crops_path = self._prepare_crops_dir(crops_dir)

        # Process each frame that has bounding boxes
        for frame_key, frame_bboxes in bboxes.items():
            img_path = None

            if frame_key in image_files:
                img_path = image_files[frame_key]
            elif int(frame_key) in image_files:
                img_path = image_files[int(frame_key)]
            elif str(frame_key) in image_files:
                img_path = image_files[str(frame_key)]
            else:
                for key, path in image_files.items():
                    if str(frame_key) in str(path.stem):
                        img_path = path
                        break

            if img_path is None:
                continue

            # Infer timestamp in seconds for this frame/image
            # Zip of images: each image = 1-second timestamp
            timestamp_seconds = 0.0
            try:
                # If frame_key is numeric like 0, 1, 2, 3...
                timestamp_seconds = float(frame_key)
            except Exception:
                try:
                    stem = img_path.stem
                    match = re.search(r'(\d+)', stem)
                    if match:
                        timestamp_seconds = float(int(match.group(1)))
                    else:
                        timestamp_seconds = float(len(detections))
                except Exception:
                    timestamp_seconds = float(len(detections))

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            for bbox_data in frame_bboxes:
                bbox = bbox_data['bbox']
                person_id = bbox_data['person_id']

                crop = self.crop_person(frame, bbox)
                if crop is None:
                    continue

                crop_rel_path = None
                if crops_path is not None:
                    crop_filename = f"{camera_location}_{img_path.stem}_det{len(detections):06d}.jpg"
                    full_path = crops_path / crop_filename
                    cv2.imwrite(str(full_path), crop)
                    crop_rel_path = f"{crops_path.name}/{crop_filename}"

                detections.append({
                    'detection_id': len(detections),
                    'frame_number': frame_key,
                    'image_name': img_path.name,
                    'timestamp_seconds': float(timestamp_seconds),
                    'camera_location': camera_location,
                    'bbox': bbox,
                    'person_id': person_id,
                    'crop_path': crop_rel_path
                })

        total_images = len(image_files)

        # Convention: each image = 1 second of original video
        duration_seconds = float(total_images)
        fps = 1.0  # one frame per second

        return {
            'detections': detections,
            'video_info': {
                'duration_seconds': duration_seconds,
                'fps': fps,
                'total_detections': len(detections),
                'source_type': 'images',
                'total_images': total_images
            }
        }


def create_export_files(entrance_data, exit_data, output_dir, processing_seconds):
    """
    Create CSV and JSON export files; crops are already saved as images.

    processing_seconds = total time the system took to process the request.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_created = []

    # Entrance CSV
    if entrance_data:
        entrance_df = pd.DataFrame([
            {
                'detection_id': d['detection_id'],
                'frame_number': d['frame_number'],
                'timestamp_seconds': d['timestamp_seconds'],
                'camera_location': d['camera_location'],
                'bbox_x': d['bbox'][0],
                'bbox_y': d['bbox'][1],
                'bbox_width': d['bbox'][2],
                'bbox_height': d['bbox'][3],
                'person_id': d['person_id'],
                'crop_path': d.get('crop_path')
            }
            for d in entrance_data['detections']
        ])

        entrance_csv = output_dir / 'entrance_detections.csv'
        entrance_df.to_csv(entrance_csv, index=False)
        files_created.append(str(entrance_csv))

    # Exit CSV
    if exit_data:
        exit_df = pd.DataFrame([
            {
                'detection_id': d['detection_id'],
                'frame_number': d['frame_number'],
                'timestamp_seconds': d['timestamp_seconds'],
                'camera_location': d['camera_location'],
                'bbox_x': d['bbox'][0],
                'bbox_y': d['bbox'][1],
                'bbox_width': d['bbox'][2],
                'bbox_height': d['bbox'][3],
                'person_id': d['person_id'],
                'crop_path': d.get('crop_path')
            }
            for d in exit_data['detections']
        ])

        exit_csv = output_dir / 'exit_detections.csv'
        exit_df.to_csv(exit_csv, index=False)
        files_created.append(str(exit_csv))

    # Combined JSON (no heavy crops inside)
    combined_data = {
        'entrance': {
            'detections': entrance_data['detections'] if entrance_data else [],
            'video_info': entrance_data['video_info'] if entrance_data else {}
        },
        'exit': {
            'detections': exit_data['detections'] if exit_data else [],
            'video_info': exit_data['video_info'] if exit_data else {}
        },
        'summary': {
            'total_entrance_detections': len(entrance_data['detections']) if entrance_data else 0,
            'total_exit_detections': len(exit_data['detections']) if exit_data else 0,
            'processed_at': datetime.now().isoformat(),
            'processing_seconds': float(processing_seconds)
        }
    }

    combined_json = output_dir / 'combined_data.json'
    with open(combined_json, 'w') as f:
        json.dump(combined_data, f, indent=2)
    files_created.append(str(combined_json))

    # Human-readable processing time string
    processing_str = f"{processing_seconds:.2f} seconds"
    if processing_seconds >= 60:
        minutes = int(processing_seconds // 60)
        seconds = processing_seconds % 60
        processing_str = f"{minutes} min {seconds:.1f} s"

    # Summary text
    summary_txt = output_dir / 'summary.txt'
    with open(summary_txt, 'w') as f:
        f.write("Re-ID Dataset Processing Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Processing time (end-to-end): {processing_str}\n\n")

        if entrance_data:
            f.write("Entrance Camera:\n")
            f.write(f"  Detections: {len(entrance_data['detections'])}\n")
            f.write(
                f"  Source duration (video/images): "
                f"{entrance_data['video_info']['duration_seconds']:.1f}s\n"
            )
            f.write(
                f"  Source FPS (or 1.0 for ZIP images): "
                f"{entrance_data['video_info']['fps']:.2f}\n\n"
            )

        if exit_data:
            f.write("Exit Camera:\n")
            f.write(f"  Detections: {len(exit_data['detections'])}\n")
            f.write(
                f"  Source duration (video/images): "
                f"{exit_data['video_info']['duration_seconds']:.1f}s\n"
            )
            f.write(
                f"  Source FPS (or 1.0 for ZIP images): "
                f"{exit_data['video_info']['fps']:.2f}\n\n"
            )

    files_created.append(str(summary_txt))

    return files_created


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


@app.route('/process', methods=['POST'])
def process_videos():
    """Process uploaded videos/images and bounding boxes"""
    # Start timing as soon as request hits this endpoint
    start_time = time.perf_counter()

    try:
        # Check files
        if 'entrance_video' not in request.files or 'entrance_bboxes' not in request.files:
            return jsonify({'error': 'Entrance video/images and bounding boxes are required'}), 400

        entrance_media = request.files['entrance_video']
        entrance_bboxes = request.files['entrance_bboxes']
        exit_media = request.files.get('exit_video')
        exit_bboxes = request.files.get('exit_bboxes')

        # Validate bounding boxes
        if not allowed_file(entrance_bboxes.filename, ALLOWED_JSON_EXTENSIONS):
            return jsonify({'error': 'Bounding boxes must be JSON'}), 400

        # Create temp directory for this session
        session_dir = Path(tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER']))

        # Where outputs (csv/json + crops) go
        output_dir = session_dir / 'output'
        output_dir.mkdir(exist_ok=True)
        entrance_crops_dir = output_dir / 'entrance_crops'
        exit_crops_dir = output_dir / 'exit_crops'

        # Save bounding boxes
        entrance_bboxes_path = session_dir / secure_filename(entrance_bboxes.filename)
        entrance_bboxes.save(str(entrance_bboxes_path))

        processor = SimpleReIDProcessor()

        # Process entrance
        entrance_data = None
        entrance_media_path = session_dir / secure_filename(entrance_media.filename)
        entrance_media.save(str(entrance_media_path))

        if allowed_file(entrance_media.filename, ALLOWED_VIDEO_EXTENSIONS):
            entrance_data = processor.process_video(
                str(entrance_media_path),
                str(entrance_bboxes_path),
                'entrance',
                crops_dir=str(entrance_crops_dir)
            )
        elif allowed_file(entrance_media.filename, ALLOWED_ARCHIVE_EXTENSIONS):
            import zipfile
            extract_dir = session_dir / 'entrance_images'
            extract_dir.mkdir(exist_ok=True)

            with zipfile.ZipFile(str(entrance_media_path), 'r') as zip_ref:
                zip_ref.extractall(str(extract_dir))

            entrance_data = processor.process_images(
                str(extract_dir),
                str(entrance_bboxes_path),
                'entrance',
                crops_dir=str(entrance_crops_dir)
            )
        else:
            return jsonify({'error': 'Entrance must be video (MP4/AVI/MOV/MKV) or images (ZIP)'}), 400

        # Process exit if provided
        exit_data = None
        if exit_media and exit_bboxes:
            if not allowed_file(exit_bboxes.filename, ALLOWED_JSON_EXTENSIONS):
                return jsonify({'error': 'Exit bounding boxes must be JSON'}), 400

            exit_bboxes_path = session_dir / secure_filename(exit_bboxes.filename)
            exit_bboxes.save(str(exit_bboxes_path))

            exit_media_path = session_dir / secure_filename(exit_media.filename)
            exit_media.save(str(exit_media_path))

            if allowed_file(exit_media.filename, ALLOWED_VIDEO_EXTENSIONS):
                exit_data = processor.process_video(
                    str(exit_media_path),
                    str(exit_bboxes_path),
                    'exit',
                    crops_dir=str(exit_crops_dir)
                )
            elif allowed_file(exit_media.filename, ALLOWED_ARCHIVE_EXTENSIONS):
                import zipfile
                extract_dir = session_dir / 'exit_images'
                extract_dir.mkdir(exist_ok=True)

                with zipfile.ZipFile(str(exit_media_path), 'r') as zip_ref:
                    zip_ref.extractall(str(extract_dir))

                exit_data = processor.process_images(
                    str(extract_dir),
                    str(exit_bboxes_path),
                    'exit',
                    crops_dir=str(exit_crops_dir)
                )
            else:
                return jsonify({'error': 'Exit must be video (MP4/AVI/MOV/MKV) or images (ZIP)'}), 400

        # Compute processing time
        processing_seconds = time.perf_counter() - start_time

        # Create export files (CSV/JSON/SUMMARY) with processing time
        files_created = create_export_files(entrance_data, exit_data, output_dir, processing_seconds)

        # Create zip file of everything under output/
        zip_path = session_dir / 'reid_dataset.zip'
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', output_dir)

        # Clean up uploaded files (keep only output)
        entrance_media_path.unlink(missing_ok=True)
        entrance_bboxes_path.unlink(missing_ok=True)
        if exit_media and exit_bboxes:
            exit_media_path.unlink(missing_ok=True)
            exit_bboxes_path.unlink(missing_ok=True)

        return jsonify({
            'success': True,
            'download_url': f'/download/{session_dir.name}',
            'entrance_detections': len(entrance_data['detections']),
            'exit_detections': len(exit_data['detections']) if exit_data else 0
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<session_id>')
def download(session_id):
    """Download processed results"""
    session_dir = Path(app.config['UPLOAD_FOLDER']) / session_id
    zip_path = session_dir / 'reid_dataset.zip'

    if not zip_path.exists():
        return jsonify({'error': 'File not found'}), 404

    return send_file(
        str(zip_path),
        as_attachment=True,
        download_name='reid_dataset.zip',
        mimetype='application/zip'
    )


if __name__ == '__main__':
    # Local development
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # For gunicorn
    pass
