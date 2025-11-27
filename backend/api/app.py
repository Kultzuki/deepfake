"""
Flask API for Deepfake Detection
Provides REST endpoints for frontend integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import sys
import tempfile
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import asyncio
from realitydefender import RealityDefender
import base64
from io import BytesIO
from PIL import Image as PILImage

# Add detection module to path
sys.path.append(str(Path(__file__).parent.parent))
from detection.detector import DeepfakeDetector

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize detector
print("Loading deepfake detector...")
detector = DeepfakeDetector(device='cuda')
print("✅ Detector ready!")

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'webm', 'avi', 'mkv', 'png', 'jpg', 'jpeg', 'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image(filename):
    return filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def is_video(filename):
    return filename.rsplit('.', 1)[1].lower() in {'mp4', 'mov', 'webm', 'avi', 'mkv'}

def is_audio(filename):
    return filename.rsplit('.', 1)[1].lower() in {'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg'}

async def analyze_with_reality_defender_async(file_path, api_key):
    """
    Analyze file using external deepfake detection API (async).
    Returns score (0-1) or None if failed.
    Supports: Images and Audio only
    """
    try:
        print(f"🔍 Detecting...")
        
        # Initialize the SDK
        rd = RealityDefender(api_key=api_key)
        
        # Upload file for analysis
        response = await rd.upload(file_path=file_path)
        request_id = response["request_id"]
        
        # Get results by polling
        result = await rd.get_result(request_id)
        print(f"✅ Detection complete!")
        
        # Extract score from result
        status = result.get('status', '')
        score = result.get('score')
        
        if score is not None:
            # Use score directly as fake probability
            fake_probability = float(score)
            
            print(f"📊 Score: {fake_probability:.3f} | Verdict: {'FAKE' if fake_probability > 0.5 else 'REAL'}")
            return fake_probability
        
        print(f"⚠️  Detection failed")
        return None
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return None

def analyze_with_reality_defender_sync(file_path, api_key):
    """
    Synchronous wrapper for async external API analysis.
    """
    try:
        # Create new event loop for this operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            analyze_with_reality_defender_async(file_path, api_key)
        )
        loop.close()
        return result
    except Exception as e:
        print(f"❌ API wrapper error: {e}")
        return None


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'detector': 'ready',
        'models': ['GANDCTAnalysis', 'OpenCLIP']
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image or video"""
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: MP4, MOV, WebM, AVI, MKV, PNG, JPG, JPEG, MP3, WAV, FLAC, AAC, M4A, OGG'}), 400
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Determine file type and analyze
        filename = secure_filename(file.filename)
        
        if is_image(filename):
            # Image analysis
            image = cv2.imread(tmp_path)
            if image is None:
                os.unlink(tmp_path)
                return jsonify({'error': 'Failed to load image'}), 400
            
            # Get image metadata
            height, width = image.shape[:2]
            file_size = os.path.getsize(tmp_path)
            
            # Image detection
            print(f"📷 Analyzing image: {filename}")
            rd_score = analyze_with_reality_defender_sync(
                tmp_path, 
                'rd_6345d8a05f4f6bac_3ed436607bd550995ec3b3abf764afcb'
            )
            
            if rd_score is None:
                # Detection failed
                os.unlink(tmp_path)
                return jsonify({'error': 'Image analysis service temporarily unavailable. Please try again.'}), 503
            
            # Only generate heatmap for fake images
            heatmap_base64 = None
            if rd_score >= 0.5:  # Fake image
                print(f"🗺️  Generating heatmap...")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                heatmap = detector.generate_heatmap(image_rgb)
                
                # Enhance variations for fake images
                heatmap = np.clip(heatmap * (0.5 + rd_score), 0, 1)
                
                # Convert heatmap to colored overlay (red=suspicious, blue=normal)
                heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                # Resize to smaller size for display (max 400px width)
                display_width = min(400, width)
                display_height = int(height * (display_width / width))
                heatmap_resized = cv2.resize(heatmap_colored, (display_width, display_height))
                
                # Encode heatmap to base64
                pil_heatmap = PILImage.fromarray(heatmap_resized)
                buffered = BytesIO()
                pil_heatmap.save(buffered, format="PNG")
                heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                print(f"✅ Heatmap generated!")
            else:
                print(f"ℹ️  Skipping heatmap for real image")
            
            # rd_score is now the fake probability (0-1, already inverted)
            # Convert to 0-100 for frontend
            fake_prob = int(rd_score * 100)
            
            # Calculate confidence based on score extremity
            # For FAKE (score > 0.5): confidence is how close to 100%
            # For REAL (score < 0.5): confidence is how close to 0%
            if rd_score > 0.5:
                # FAKE verdict: confidence = fake probability
                confidence = fake_prob
            else:
                # REAL verdict: confidence = real probability (100 - fake_prob)
                confidence = 100 - fake_prob
            
            response = {
                'verdict': 'FAKE' if rd_score > 0.5 else 'REAL',
                'confidence': confidence,
                'source': 'cloud_api',  # Flag for frontend to hide model breakdown
                'heatmap': f"data:image/png;base64,{heatmap_base64}" if heatmap_base64 else None,
                'scores': {
                    # External API provides a single combined score
                    # We don't have separate appearance/frequency from cloud API
                    # Use the combined score for visualization purposes
                    'appearance': fake_prob,
                    'frequency': fake_prob,
                    'final': fake_prob
                },
                'metadata': {
                    'type': 'image',
                    'resolution': f"{width}x{height}",
                    'fileSize': file_size,
                    'format': filename.rsplit('.', 1)[1].upper()
                },
                'warnings': []
            }
        
        elif is_audio(filename):
            # Audio analysis
            print(f"🎵 Analyzing audio: {filename}")
            rd_score = analyze_with_reality_defender_sync(
                tmp_path, 
                'rd_6345d8a05f4f6bac_3ed436607bd550995ec3b3abf764afcb'
            )
            
            if rd_score is None:
                # Detection failed
                os.unlink(tmp_path)
                return jsonify({'error': 'Audio analysis service temporarily unavailable. Please try again.'}), 503
            
            # Get audio metadata
            file_size = os.path.getsize(tmp_path)
            
            # Convert score to percentage
            fake_prob = int(rd_score * 100)
            
            # Calculate confidence
            if rd_score > 0.5:
                # FAKE verdict: confidence = fake probability
                confidence = fake_prob
            else:
                # REAL verdict: confidence = real probability (100 - fake_prob)
                confidence = 100 - fake_prob
            
            response = {
                'verdict': 'FAKE' if rd_score > 0.5 else 'REAL',
                'confidence': confidence,
                'source': 'cloud_api',
                'heatmap': None,  # No heatmap for audio
                'scores': {
                    'appearance': fake_prob,
                    'frequency': fake_prob,
                    'final': fake_prob
                },
                'metadata': {
                    'type': 'audio',
                    'fileSize': file_size,
                    'format': filename.rsplit('.', 1)[1].upper()
                },
                'warnings': []
            }
        
        elif is_video(filename):
            # Video analysis
            print(f"🎬 Analyzing video: {filename}")
            result = detector.predict_video(tmp_path, num_frames=15)
            
            if result is None:
                os.unlink(tmp_path)
                return jsonify({'error': 'Failed to process video'}), 400
            
            # Get video metadata
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            file_size = os.path.getsize(tmp_path)
            
            # Determine verdict first
            method = result.get('method', 'clip_dct_fallback')
            final_fake_prob = int(result.get('mean_hf_score', result.get('mean_score', 0)) * 100)
            verdict = 'FAKE' if final_fake_prob > 50 else 'REAL'
            
            # Extract 5 frames and generate heatmaps for fake videos
            video_frames = []
            video_heatmaps = []
            
            if verdict == 'FAKE':
                print(f"🗺️  Generating frame previews and heatmaps for fake video...")
                frame_indices = np.linspace(0, frame_count - 1, 5, dtype=int)
                
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Convert frame to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Resize frame for preview (max 400px width)
                        display_width = min(400, width)
                        display_height = int(height * (display_width / width))
                        frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
                        
                        # Encode frame to base64
                        pil_frame = PILImage.fromarray(frame_resized)
                        buffered = BytesIO()
                        pil_frame.save(buffered, format="JPEG")
                        frame_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        video_frames.append(f"data:image/jpeg;base64,{frame_base64}")
                        
                        # Generate heatmap for this frame
                        heatmap = detector.generate_heatmap(frame_rgb)
                        heatmap = np.clip(heatmap * (0.5 + final_fake_prob / 100), 0, 1)
                        
                        # Convert heatmap to colored overlay
                        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                        heatmap_resized = cv2.resize(heatmap_colored, (display_width, display_height))
                        
                        # Encode heatmap to base64
                        pil_heatmap = PILImage.fromarray(heatmap_resized)
                        buffered = BytesIO()
                        pil_heatmap.save(buffered, format="PNG")
                        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        video_heatmaps.append(f"data:image/png;base64,{heatmap_base64}")
                
                print(f"✅ Generated {len(video_frames)} frame previews with heatmaps!")
            else:
                print(f"ℹ️  Skipping heatmaps for real video")
            
            cap.release()
            
            # Format duration
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            duration_str = f"{minutes:02d}:{seconds:02d}"
            
            # DEBUG: Print raw scores based on method
            print(f"📊 Raw Video Scores (Method: {method}):")
            
            if method == 'huggingface':
                # HuggingFace model was used
                print(f"   HF Score: {result['mean_hf_score']:.3f}")
                print(f"   Final Score: {result['mean_score']:.3f}")
                print(f"   Prediction: {result['prediction']}")
                
                # Convert to 0-100 for frontend
                hf_fake_prob = int(result['mean_hf_score'] * 100)
                final_fake_prob = int(result['mean_score'] * 100)
                
                # Calculate confidence
                if result['prediction'] == 'FAKE':
                    confidence = final_fake_prob
                else:
                    confidence = 100 - final_fake_prob
                
                print(f"📊 Frontend Display:")
                print(f"   Verdict: {result['prediction']}")
                print(f"   Confidence: {confidence}%")
                print(f"   Detection Score: {hf_fake_prob}%")
                
                response = {
                    'verdict': result['prediction'],
                    'confidence': confidence,
                    'videoFrames': video_frames if video_frames else None,
                    'videoHeatmaps': video_heatmaps if video_heatmaps else None,
                    'scores': {
                        'appearance': hf_fake_prob,
                        'frequency': hf_fake_prob,
                        'final': final_fake_prob
                    },
                    'metadata': {
                        'type': 'video',
                        'resolution': f"{width}x{height}",
                        'duration': duration_str,
                        'fps': int(fps),
                        'codec': 'H.264',
                        'audioPresent': False,
                        'compressionLevel': 'Medium',
                        'framesAnalyzed': result['num_frames'],
                        'fileSize': file_size
                    },
                    'warnings': []
                }
            
            else:
                # Fallback mode (CLIP + DCT)
                print(f"   CLIP Score: {result['mean_clip_score']:.3f}")
                print(f"   DCT Score: {result['mean_dct_score']:.3f}")
                print(f"   Final Score: {result['mean_score']:.3f}")
                print(f"   Prediction: {result['prediction']}")
                
                # Convert to 0-100 for frontend
                clip_fake_prob = int(result['mean_clip_score'] * 100)
                dct_fake_prob = int(result['mean_dct_score'] * 100)
                final_fake_prob = int(result['mean_score'] * 100)
                
                # Calculate confidence
                if result['prediction'] == 'FAKE':
                    confidence = final_fake_prob
                else:
                    confidence = 100 - final_fake_prob
                
                print(f"📊 Frontend Display:")
                print(f"   Verdict: {result['prediction']}")
                print(f"   Confidence: {confidence}%")
                print(f"   Appearance: {clip_fake_prob}%")
                print(f"   Frequency: {dct_fake_prob}%")
                
                response = {
                    'verdict': result['prediction'],
                    'confidence': confidence,
                    'videoFrames': video_frames if video_frames else None,
                    'videoHeatmaps': video_heatmaps if video_heatmaps else None,
                    'scores': {
                        'appearance': clip_fake_prob,
                        'frequency': dct_fake_prob,
                        'final': final_fake_prob
                    },
                    'metadata': {
                        'type': 'video',
                        'resolution': f"{width}x{height}",
                        'duration': duration_str,
                        'fps': int(fps),
                        'codec': 'H.264',
                        'audioPresent': False,
                        'compressionLevel': 'Medium',
                        'framesAnalyzed': result['num_frames'],
                        'fileSize': file_size
                    },
                    'warnings': []
                }
        
        else:
            os.unlink(tmp_path)
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return jsonify(response)
    
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple files (future feature)"""
    return jsonify({'error': 'Batch analysis not yet implemented'}), 501


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  DEEPFAKE DETECTION API SERVER")
    print("="*70)
    print("\nEndpoints:")
    print("  GET  /api/health         - Health check")
    print("  POST /api/analyze        - Analyze image/video")
    print("  POST /api/batch-analyze  - Batch analysis (not implemented)")
    print("\nStarting server on http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
