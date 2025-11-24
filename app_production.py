"""
Production-ready Flask Web Application for Simple Re-ID Dataset Builder
Optimized for deployment on free hosting services
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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB for free tier
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', tempfile.mkdtemp())

# Create upload folder if it doesn't exist
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_JSON_EXTENSIONS = {'json'}


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
    
    def process_video(self, video_path, bboxes_file, camera_location):
        """Process single video"""
        bboxes = self.load_bounding_boxes(bboxes_file)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        frames_to_process = sorted(bboxes.keys())
        
        detections = []
        current_frame = 0
        
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
                
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_normalized = crop_rgb.astype(np.float32) / 255.0
                
                detections.append({
                    'detection_id': len(detections),
                    'frame_number': int(target_frame),
                    'timestamp_seconds': float(timestamp_seconds),
                    'camera_location': camera_location,
                    'bbox': bbox,
                    'person_id': person_id,
                    'crop': crop_normalized
                })
            
            current_frame += 1
        
        cap.release()
        
        return {
            'detections': detections,
            'video_info': {
                'duration_seconds': duration,
                'fps': video_fps,
                'total_detections': len(detections)
            }
        }


def create_export_files(entrance_data, exit_data, output_dir):
    """Create CSV and JSON export files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_created = []
    
    # Create CSV for entrance
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
                'person_id': d['person_id']
            }
            for d in entrance_data['detections']
        ])
        
        entrance_csv = output_dir / 'entrance_detections.csv'
        entrance_df.to_csv(entrance_csv, index=False)
        files_created.append(str(entrance_csv))
    
    # Create CSV for exit
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
                'person_id': d['person_id']
            }
            for d in exit_data['detections']
        ])
        
        exit_csv = output_dir / 'exit_detections.csv'
        exit_df.to_csv(exit_csv, index=False)
        files_created.append(str(exit_csv))
    
    # Create combined JSON
    combined_data = {
        'entrance': {
            'detections': [
                {k: v for k, v in d.items() if k != 'crop'}
                for d in entrance_data['detections']
            ] if entrance_data else [],
            'video_info': entrance_data['video_info'] if entrance_data else {}
        },
        'exit': {
            'detections': [
                {k: v for k, v in d.items() if k != 'crop'}
                for d in exit_data['detections']
            ] if exit_data else [],
            'video_info': exit_data['video_info'] if exit_data else {}
        },
        'summary': {
            'total_entrance_detections': len(entrance_data['detections']) if entrance_data else 0,
            'total_exit_detections': len(exit_data['detections']) if exit_data else 0,
            'processed_at': datetime.now().isoformat()
        }
    }
    
    combined_json = output_dir / 'combined_data.json'
    with open(combined_json, 'w') as f:
        json.dump(combined_data, f, indent=2)
    files_created.append(str(combined_json))
    
    # Save person crops as NPZ
    if entrance_data and entrance_data['detections']:
        entrance_crops = np.array([d['crop'] for d in entrance_data['detections']])
        entrance_npz = output_dir / 'entrance_person_crops.npz'
        np.savez_compressed(entrance_npz, person_crops=entrance_crops)
        files_created.append(str(entrance_npz))
    
    if exit_data and exit_data['detections']:
        exit_crops = np.array([d['crop'] for d in exit_data['detections']])
        exit_npz = output_dir / 'exit_person_crops.npz'
        np.savez_compressed(exit_npz, person_crops=exit_crops)
        files_created.append(str(exit_npz))
    
    # Create summary text
    summary_txt = output_dir / 'summary.txt'
    with open(summary_txt, 'w') as f:
        f.write("Re-ID Dataset Processing Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if entrance_data:
            f.write(f"Entrance Camera:\n")
            f.write(f"  Detections: {len(entrance_data['detections'])}\n")
            f.write(f"  Duration: {entrance_data['video_info']['duration_seconds']:.1f}s\n")
            f.write(f"  FPS: {entrance_data['video_info']['fps']:.2f}\n\n")
        
        if exit_data:
            f.write(f"Exit Camera:\n")
            f.write(f"  Detections: {len(exit_data['detections'])}\n")
            f.write(f"  Duration: {exit_data['video_info']['duration_seconds']:.1f}s\n")
            f.write(f"  FPS: {exit_data['video_info']['fps']:.2f}\n\n")
    
    files_created.append(str(summary_txt))
    
    return files_created


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint for monitoring"""
    return jsonify({'status': 'healthy'}), 200


@app.route('/process', methods=['POST'])
def process_videos():
    """Process uploaded videos and bounding boxes"""
    try:
        # Check files
        if 'entrance_video' not in request.files or 'entrance_bboxes' not in request.files:
            return jsonify({'error': 'Entrance video and bounding boxes are required'}), 400
        if 'exit_video' not in request.files or 'exit_bboxes' not in request.files:
            return jsonify({'error': 'Exit video and bounding boxes are required'}), 400
        
        entrance_video = request.files['entrance_video']
        entrance_bboxes = request.files['entrance_bboxes']
        exit_video = request.files.get('exit_video')
        exit_bboxes = request.files.get('exit_bboxes')
        
        # Validate files
        if not allowed_file(entrance_video.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({'error': 'Invalid entrance video format'}), 400
        
        if not allowed_file(entrance_bboxes.filename, ALLOWED_JSON_EXTENSIONS):
            return jsonify({'error': 'Bounding boxes must be JSON'}), 400
        
        # Create temp directory for this session
        session_dir = Path(tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER']))
        
        # Save uploaded files
        entrance_video_path = session_dir / secure_filename(entrance_video.filename)
        entrance_bboxes_path = session_dir / secure_filename(entrance_bboxes.filename)
        
        entrance_video.save(str(entrance_video_path))
        entrance_bboxes.save(str(entrance_bboxes_path))
        
        # Process entrance
        processor = SimpleReIDProcessor()
        entrance_data = processor.process_video(
            str(entrance_video_path),
            str(entrance_bboxes_path),
            'entrance'
        )
        
        # Process exit if provided
        exit_data = None
        if exit_video and exit_bboxes:
            if not allowed_file(exit_video.filename, ALLOWED_VIDEO_EXTENSIONS):
                return jsonify({'error': 'Invalid exit video format'}), 400
            
            if not allowed_file(exit_bboxes.filename, ALLOWED_JSON_EXTENSIONS):
                return jsonify({'error': 'Exit bounding boxes must be JSON'}), 400
            
            exit_video_path = session_dir / secure_filename(exit_video.filename)
            exit_bboxes_path = session_dir / secure_filename(exit_bboxes.filename)
            
            exit_video.save(str(exit_video_path))
            exit_bboxes.save(str(exit_bboxes_path))
            
            exit_data = processor.process_video(
                str(exit_video_path),
                str(exit_bboxes_path),
                'exit'
            )
        
        # Create export files
        output_dir = session_dir / 'output'
        files_created = create_export_files(entrance_data, exit_data, output_dir)
        
        # Create zip file
        zip_path = session_dir / 'reid_dataset.zip'
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', output_dir)
        
        # Clean up video files to save space
        entrance_video_path.unlink()
        entrance_bboxes_path.unlink()
        if exit_video and exit_bboxes:
            exit_video_path.unlink()
            exit_bboxes_path.unlink()
        
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
    app.run(debug=True, host='0.0.0.0', port=5005)
else:
    # Production (for Gunicorn)
    pass