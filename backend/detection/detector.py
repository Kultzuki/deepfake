"""
Visual-Only Deepfake Detector
Combines HuggingFace pre-trained model (primary), CLIP, and DCT analysis (fallbacks).
No audio required.
"""

import sys
from pathlib import Path

# Add models to path
sys.path.append(str(Path(__file__).parent.parent / "models" / "GANDCTAnalysis"))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import open_clip
from scipy import fftpack
import pandas as pd
from tqdm import tqdm
import argparse
from transformers import AutoImageProcessor, AutoModelForImageClassification


from transformers import AutoImageProcessor, AutoModelForImageClassification


class HuggingFaceDetector:
    """Pre-trained deepfake detector from Hugging Face."""
    
    def __init__(self, model_path=None, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Use local model path or fallback to hub
        if model_path is None:
            model_path = str(Path(__file__).parent.parent.parent / "models" / "deepfake_detector_v1")
        
        print(f"Loading HuggingFace model from {model_path}...")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForImageClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ HuggingFace Detector initialized on {self.device}")
            self.available = True
        except Exception as e:
            print(f"⚠️ Failed to load HuggingFace model: {e}")
            self.available = False
    
    def predict(self, image):
        """Predict if image is fake. Returns score 0-1 (0=real, 1=fake)."""
        if not self.available:
            return None
        
        try:
            # Convert to PIL if needed
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                image = Image.fromarray(image)
            
            # Process and predict
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Assuming labels: [REAL, FAKE] or similar
            # Get fake probability (usually index 1)
            fake_prob = probs[0][1].item() if probs.shape[1] > 1 else probs[0][0].item()
            
            return float(np.clip(fake_prob, 0, 1))
        
        except Exception as e:
            print(f"⚠️ HuggingFace prediction failed: {e}")
            return None
    
    def generate_heatmap(self, image, target_size=(224, 224)):
        """
        Generate heatmap using patch-based analysis with HuggingFace model.
        Returns heatmap as numpy array (H, W) with values 0-1.
        """
        if not self.available:
            return None
        
        try:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                orig_image = Image.fromarray(image)
            else:
                orig_image = image
            
            orig_width, orig_height = orig_image.size
            
            # Create patch-based heatmap
            patch_size = 64  # Larger patches for efficiency
            stride = 32      # Move 32 pixels at a time
            
            heatmap = np.zeros((orig_height // stride, orig_width // stride))
            
            with torch.no_grad():
                for i, y in enumerate(range(0, orig_height - patch_size, stride)):
                    for j, x in enumerate(range(0, orig_width - patch_size, stride)):
                        # Extract patch
                        patch = orig_image.crop((x, y, x + patch_size, y + patch_size))
                        
                        # Get prediction for patch
                        patch_score = self.predict(patch)
                        
                        if patch_score is not None and i < heatmap.shape[0] and j < heatmap.shape[1]:
                            heatmap[i, j] = patch_score
            
            # Resize heatmap to target size and normalize
            heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
            heatmap_resized = np.clip(heatmap_resized, 0, 1)
            
            return heatmap_resized
        
        except Exception as e:
            print(f"⚠️ HuggingFace heatmap generation failed: {e}")
            return None


class DCTAnalyzer:
    """Frequency domain analysis using DCT."""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"DCT Analyzer initialized on {self.device}")
    
    def compute_dct(self, image):
        """Compute DCT of image."""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        image = np.array(image, dtype=np.uint8)
        
        if len(image.shape) == 3:
            image_ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            image_gray = image_ycbcr[:, :, 0]
        else:
            image_gray = image
        
        dct = fftpack.dct(fftpack.dct(image_gray.T, norm='ortho').T, norm='ortho')
        return dct
    
    def analyze_dct_statistics(self, dct_coeffs):
        """Analyze DCT statistics for GAN artifacts."""
        h, w = dct_coeffs.shape
        high_freq = dct_coeffs[h//2:, w//2:]
        
        mean_high_freq = np.abs(high_freq).mean()
        std_high_freq = np.abs(high_freq).std()
        
        score = np.tanh(mean_high_freq / 10.0 + std_high_freq / 20.0)
        return float(score)
    
    def predict(self, image):
        """Predict if image is fake."""
        image_resized = cv2.resize(image, (128, 128))
        dct_coeffs = self.compute_dct(image_resized)
        score = self.analyze_dct_statistics(dct_coeffs)
        return score


class CLIPAnalyzer:
    """Semantic analysis using OpenCLIP."""
    
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        texts = [
            "a photo of a real person",
            "a photo of a real human face",
            "an authentic photograph",
            "a deepfake image",
            "a synthetic face",
            "an AI-generated image"
        ]
        
        with torch.no_grad():
            text_tokens = self.tokenizer(texts).to(self.device)
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
        self.real_features = self.text_features[:3].mean(dim=0, keepdim=True)
        self.fake_features = self.text_features[3:].mean(dim=0, keepdim=True)
        
        print(f"✅ CLIP Analyzer initialized on {self.device}")
    
    def predict(self, image):
        """Predict if image is fake."""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            image = Image.fromarray(image)
        
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        real_sim = (image_features @ self.real_features.T).item()
        fake_sim = (image_features @ self.fake_features.T).item()
        
        # Normalize similarities to [0, 1] range and compute fake probability
        # Higher fake_sim relative to real_sim = higher fake probability
        total_sim = abs(real_sim) + abs(fake_sim)
        if total_sim > 0:
            score = abs(fake_sim) / total_sim
        else:
            score = 0.5  # Neutral if both similarities are zero
        
        score = np.clip(score, 0, 1)
        
        return float(score)
    
    def generate_heatmap(self, image, target_size=(224, 224)):
        """
        Generate heatmap showing suspicious regions using patch-based analysis.
        Returns heatmap as numpy array (H, W) with values 0-1.
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            orig_image = Image.fromarray(image)
        else:
            orig_image = image
        
        orig_width, orig_height = orig_image.size
        
        # Create patch-based heatmap
        patch_size = 32  # Analyze 32x32 patches
        stride = 16      # Move 16 pixels at a time for overlap
        
        heatmap = np.zeros((orig_height // stride, orig_width // stride))
        
        with torch.no_grad():
            for i, y in enumerate(range(0, orig_height - patch_size, stride)):
                for j, x in enumerate(range(0, orig_width - patch_size, stride)):
                    # Extract patch
                    patch = orig_image.crop((x, y, x + patch_size, y + patch_size))
                    patch_tensor = self.preprocess(patch).unsqueeze(0).to(self.device)
                    
                    # Get features
                    patch_features = self.model.encode_image(patch_tensor)
                    patch_features /= patch_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate fake similarity
                    fake_sim = (patch_features @ self.fake_features.T).item()
                    real_sim = (patch_features @ self.real_features.T).item()
                    
                    # Score: higher = more fake-like
                    total_sim = abs(real_sim) + abs(fake_sim)
                    if total_sim > 0:
                        patch_score = abs(fake_sim) / total_sim
                    else:
                        patch_score = 0.5
                    
                    if i < heatmap.shape[0] and j < heatmap.shape[1]:
                        heatmap[i, j] = patch_score
        
        # Resize heatmap to target size and normalize
        heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
        heatmap_resized = np.clip(heatmap_resized, 0, 1)
        
        return heatmap_resized


class DeepfakeDetector:
    """
    Visual-only deepfake detector with fallback system.
    Priority: HuggingFace (primary) → CLIP → DCT (fallbacks).
    """
    
    def __init__(self, dct_weight=0.43, clip_weight=0.57, device='cuda'):
        self.dct_weight = dct_weight
        self.clip_weight = clip_weight
        self.device = device
        
        print("Initializing Visual Deepfake Detector...")
        
        # Initialize HuggingFace model (primary)
        self.hf_detector = HuggingFaceDetector(device=device)
        
        # Initialize fallback detectors
        self.dct_analyzer = DCTAnalyzer(device=device)
        self.clip_analyzer = CLIPAnalyzer(device=device)
        
        if self.hf_detector.available:
            print(f"✅ Detector initialized (Primary: HuggingFace, Fallback: CLIP + DCT)")
        else:
            print(f"✅ Detector initialized (CLIP + DCT fallback mode)")
    
    def predict_frame(self, image):
        """Predict single frame with fallback system."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Try HuggingFace model first (primary)
        hf_score = self.hf_detector.predict(image)
        
        if hf_score is not None:
            # HuggingFace succeeded - use it as primary
            return {
                'method': 'huggingface',
                'hf_score': hf_score,
                'final_score': hf_score,
                'prediction': 'FAKE' if hf_score > 0.5 else 'REAL',
                'confidence': abs(hf_score - 0.5) * 2
            }
        
        # Fallback to CLIP + DCT
        dct_score = self.dct_analyzer.predict(image)
        clip_score = self.clip_analyzer.predict(image)
        
        final_score = (
            self.dct_weight * dct_score +
            self.clip_weight * clip_score
        )
        
        return {
            'method': 'clip_dct_fallback',
            'dct_score': dct_score,
            'clip_score': clip_score,
            'final_score': final_score,
            'prediction': 'FAKE' if final_score > 0.5 else 'REAL',
            'confidence': abs(final_score - 0.5) * 2
        }
    
    def generate_heatmap(self, image):
        """
        Generate heatmap for an image showing suspicious regions.
        Tries HuggingFace first, falls back to CLIP.
        Returns heatmap as numpy array (H, W) with values 0-1.
        """
        # Try HuggingFace first
        heatmap = self.hf_detector.generate_heatmap(image)
        
        if heatmap is not None:
            return heatmap
        
        # Fallback to CLIP
        return self.clip_analyzer.generate_heatmap(image)
    
    def predict_video(self, video_path, num_frames=15):
        """Predict entire video."""
        cap = cv2.VideoCapture(str(video_path))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return None
        
        frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        
        frame_results = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.predict_frame(frame_rgb)
            frame_results.append(result)
        
        cap.release()
        
        if not frame_results:
            return None
        
        # Aggregate results based on method used
        method = frame_results[0].get('method', 'clip_dct_fallback')
        
        if method == 'huggingface':
            hf_scores = [r.get('hf_score', 0) for r in frame_results]
            mean_score = np.mean(hf_scores)
            max_score = np.max(hf_scores)
            std_score = np.std(hf_scores)
            
            return {
                'method': 'huggingface',
                'num_frames': len(frame_results),
                'mean_hf_score': np.mean(hf_scores),
                'mean_score': mean_score,
                'max_score': max_score,
                'std_score': std_score,
                'prediction': 'FAKE' if mean_score > 0.5 else 'REAL',
                'confidence': abs(mean_score - 0.5) * 2
            }
        else:
            # Fallback mode - CLIP + DCT
            dct_scores = [r.get('dct_score', 0) for r in frame_results]
            clip_scores = [r.get('clip_score', 0) for r in frame_results]
            final_scores = [r['final_score'] for r in frame_results]
            
            mean_score = np.mean(final_scores)
            max_score = np.max(final_scores)
            std_score = np.std(final_scores)
            
            return {
                'method': 'clip_dct_fallback',
                'num_frames': len(frame_results),
                'mean_dct_score': np.mean(dct_scores),
                'mean_clip_score': np.mean(clip_scores),
                'mean_score': mean_score,
                'max_score': max_score,
                'std_score': std_score,
                'prediction': 'FAKE' if mean_score > 0.5 else 'REAL',
                'confidence': abs(mean_score - 0.5) * 2
            }
    
    def predict_batch(self, video_paths, output_csv=None):
        """Predict batch of videos."""
        results = []
        
        for video_path in tqdm(video_paths, desc="Processing videos"):
            result = self.predict_video(video_path)
            
            if result:
                results.append({
                    'video_path': str(video_path),
                    'prediction': result['prediction'],
                    'score': result['mean_score'],
                    'confidence': result['confidence'],
                    'num_frames': result['num_frames']
                })
        
        df = pd.DataFrame(results)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\n✅ Results saved to {output_csv}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description="Visual Deepfake Detection")
    parser.add_argument('--video', type=str, help='Single video path')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--metadata', type=str, help='Metadata CSV for batch processing')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], help='Split to process')
    parser.add_argument('--output', type=str, default='results/predictions.csv', help='Output CSV path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create detector
    detector = DeepfakeDetector(device=args.device)
    
    if args.image:
        # Single image
        print(f"\n{'='*70}")
        print(f"Processing: {Path(args.image).name}")
        print(f"{'='*70}")
        
        # Load image
        image = cv2.imread(str(args.image))
        if image is None:
            print("❌ Failed to load image")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.predict_frame(image_rgb)
        
        print(f"\n{'='*70}")
        print("DETECTION RESULTS")
        print(f"{'='*70}")
        print(f"Prediction:     {result['prediction']}")
        print(f"Score:          {result['final_score']:.3f}")
        print(f"Confidence:     {result['confidence']:.3f}")
        print(f"DCT Score:      {result['dct_score']:.3f}")
        print(f"CLIP Score:     {result['clip_score']:.3f}")
        print(f"{'='*70}\n")
    
    elif args.video:
        # Single video
        print(f"\n{'='*70}")
        print(f"Processing: {Path(args.video).name}")
        print(f"{'='*70}")
        
        result = detector.predict_video(args.video)
        
        if result:
            print(f"\n{'='*70}")
            print("DETECTION RESULTS")
            print(f"{'='*70}")
            print(f"Prediction:     {result['prediction']}")
            print(f"Score:          {result['mean_score']:.3f}")
            print(f"Confidence:     {result['confidence']:.3f}")
            print(f"Frames analyzed: {result['num_frames']}")
            print(f"DCT Score:      {result['mean_dct_score']:.3f}")
            print(f"CLIP Score:     {result['mean_clip_score']:.3f}")
            print(f"{'='*70}\n")
    
    elif args.metadata:
        # Batch processing
        df = pd.read_csv(args.metadata)
        
        if args.split:
            df = df[df['split'] == args.split]
            print(f"Processing {args.split} split: {len(df)} videos")
        
        video_paths = df['video_path'].tolist()
        
        # Ensure output directory exists
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        
        # Process batch
        results_df = detector.predict_batch(video_paths, output_csv=args.output)
        
        # Merge with metadata
        full_results = df.merge(results_df, on='video_path', how='left')
        full_results.to_csv(args.output, index=False)
        
        # Calculate accuracy if labels available
        if 'label' in full_results.columns:
            full_results['predicted_label'] = (full_results['score'] > 0.5).astype(int)
            accuracy = (full_results['predicted_label'] == full_results['label']).mean()
            
            print(f"\n{'='*70}")
            print("EVALUATION METRICS")
            print(f"{'='*70}")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Total videos: {len(full_results)}")
            print(f"Real: {(full_results['label'] == 0).sum()}")
            print(f"Fake: {(full_results['label'] == 1).sum()}")
            print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
