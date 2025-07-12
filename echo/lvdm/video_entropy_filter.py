"""
PanEcho-based Video Quality Filter

This script filters generated videos based on feature entropy comparison with original training videos.
It uses the PanEcho model to extract video features and identifies outliers based on entropy distribution.

Usage:
    python video_filter.py --original_dir /path/to/original/videos --generated_dir /path/to/generated/videos

Example:
    python video_entropy_filter.py 
        --original_dir /nfs/usrhome/khmuhammad/EchoPath/datasets/CardiacASD/mp4 
        --generated_dir /nfs/scratch/EchoPath/samples/lvdm_asd_triplets/mp4
        --output_dir /nfs/usrhome/khmuhammad/EchoPath/asd_filter_results 
        --video_size 112 
        --threshold_percentile 20 
        --copy_good_videos

Requirements:
    pip install torch torchvision opencv-python scipy numpy matplotlib tqdm
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import cv2
from typing import List, Tuple, Dict, Optional
from scipy.stats import entropy
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging
import argparse
import os
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoMetrics:
    """Store metrics for a video"""
    path: str
    features: np.ndarray
    entropy_value: float
    is_good: bool = True

class PanEchoFeatureExtractor:
    """Video feature extraction using PanEcho model"""
    def __init__(self, clip_len: int = 32, video_size: int = 112, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.clip_len = clip_len
        self.video_size = video_size
        self.feature_output = None  # To store captured encoder output
        self.hook = None  # To store the hook handle
        self.model = self._load_model()
        logger.info(f"Initialized PanEcho with video_size={video_size}x{video_size}")
    
    def _hook_fn(self, module, input, output):
        """Hook function to capture encoder output"""
        self.feature_output = output
        
    def _clear_torch_hub_cache(self):
        """Clear torch hub cache to force fresh download"""
        try:
            hub_dir = torch.hub.get_dir()
            if os.path.exists(hub_dir):
                logger.info(f"Clearing torch hub cache at {hub_dir}")
                shutil.rmtree(hub_dir)
                logger.info("Torch hub cache cleared successfully")
        except Exception as e:
            logger.warning(f"Could not clear torch hub cache: {e}")
    
    def _load_model(self):
        """Load PanEcho model from torch hub with robust error handling"""
        logger.info("Loading PanEcho model...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} to load PanEcho model")
                
                # Try loading with force_reload=False first
                force_reload = attempt > 0  # Force reload on subsequent attempts
                
                if force_reload:
                    logger.info("Force reloading model due to previous failure")
                
                model = torch.hub.load(
                    'CarDS-Yale/PanEcho', 
                    'PanEcho', 
                    force_reload=force_reload,
                    clip_len=self.clip_len,
                    trust_repo=True  # Add trust_repo parameter
                )
                
                # Verify model loaded correctly
                if model is None:
                    raise RuntimeError("Model loaded but returned None")
                
                # Register hook on the encoder to capture feature extractor output
                self.hook = model.encoder.register_forward_hook(self._hook_fn)
                logger.info("Registered hook on encoder to capture feature extractor output")
                
                # Test model with dummy input to ensure it works
                dummy_input = torch.randn(1, 3, self.clip_len, 224, 224).to(self.device)
                model = model.to(self.device).eval()
                
                with torch.no_grad():
                    try:
                        test_output = model(dummy_input)
                        if test_output is None:
                            raise RuntimeError("Model forward pass returned None")
                        
                        # Verify we captured encoder features
                        if self.feature_output is None:
                            raise RuntimeError("Failed to capture encoder output via hook")
                        
                        logger.info(f"Model test successful, captured encoder features shape: {self.feature_output.shape}")
                        
                        # Reset feature_output for actual use
                        self.feature_output = None
                            
                    except Exception as e:
                        raise RuntimeError(f"Model forward pass failed: {e}")
                
                logger.info("PanEcho model loaded and verified successfully")
                return model
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                
                # Clean up hook if it was registered
                if self.hook is not None:
                    self.hook.remove()
                    self.hook = None
                
                if attempt < max_retries - 1:
                    logger.info("Clearing cache and retrying...")
                    self._clear_torch_hub_cache()
                    
                    # Also clear any potentially corrupted model files
                    try:
                        hub_dir = torch.hub.get_dir()
                        panecho_dir = os.path.join(hub_dir, "CarDS-Yale_PanEcho_main")
                        if os.path.exists(panecho_dir):
                            logger.info(f"Removing potentially corrupted PanEcho directory: {panecho_dir}")
                            shutil.rmtree(panecho_dir)
                    except Exception as cleanup_error:
                        logger.warning(f"Could not clean up PanEcho directory: {cleanup_error}")
                    
                    # Wait a bit before retrying
                    import time
                    time.sleep(2)
                else:
                    logger.error("All attempts to load PanEcho model failed")
                    
                    # Provide helpful error message
                    error_msg = (
                        f"Failed to load PanEcho model after {max_retries} attempts. "
                        "This could be due to:\n"
                        "1. Network connectivity issues\n"
                        "2. Corrupted download cache\n"
                        "3. GitHub repository access issues\n"
                        "4. Incompatible PyTorch version\n"
                        "5. Insufficient GPU memory\n\n"
                        "Troubleshooting steps:\n"
                        "- Check internet connection\n"
                        "- Manually clear torch hub cache: rm -rf ~/.cache/torch/hub/\n"
                        "- Try running with --device cpu\n"
                        "- Ensure PyTorch >= 1.8.0 is installed\n"
                        f"Original error: {e}"
                    )
                    raise RuntimeError(error_msg)
    
    def extract_features(self, video_path: str, stride: Optional[int] = None) -> np.ndarray:
        """
        Extract features from a video using PanEcho model
        
        Args:
            video_path: Path to video file
            stride: Frame stride for sampling multiple clips. If None, extracts one clip
        
        Returns:
            Feature vector as numpy array
        """
        # Load video frames
        all_frames = self._load_all_video_frames(video_path)
        
        if len(all_frames) < self.clip_len:
            logger.warning(f"Video {video_path} has fewer than {self.clip_len} frames. Padding...")
            all_frames = self._pad_frames(all_frames, self.clip_len)
        
        # Extract features from clips
        if stride is None:
            # Single clip from the middle of the video
            start_idx = max(0, (len(all_frames) - self.clip_len) // 2)
            frames = all_frames[start_idx:start_idx + self.clip_len]
            features = self._extract_clip_features(frames)
        else:
            # Multiple clips with stride
            features_list = []
            for start_idx in range(0, len(all_frames) - self.clip_len + 1, stride):
                frames = all_frames[start_idx:start_idx + self.clip_len]
                clip_features = self._extract_clip_features(frames)
                features_list.append(clip_features)
            
            # Average pooling across clips
            features = np.mean(features_list, axis=0)
        
        return features.flatten()
    
    def _extract_clip_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract features from a single clip of frames"""
        # Preprocess frames
        frames_tensor = self._preprocess_frames(frames)
        
        # Reset feature_output before forward pass
        self.feature_output = None
        
        # Extract features
        with torch.no_grad():
            output = self.model(frames_tensor)
            
        # Use the captured encoder features instead of model output
        if self.feature_output is None:
            raise RuntimeError("Failed to capture encoder features via hook")
        
        features = self.feature_output
        
        # Convert to numpy array
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        else:
            raise RuntimeError(f"Unexpected feature type from encoder: {type(features)}")
        
        return features
    
    def _load_all_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load all frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return frames
    
    def _pad_frames(self, frames: List[np.ndarray], target_len: int) -> List[np.ndarray]:
        """Pad frames by repeating the last frame"""
        while len(frames) < target_len:
            frames.append(frames[-1].copy())
        return frames
    
    def _preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess frames for PanEcho model"""
        # Ensure we have exactly clip_len frames
        assert len(frames) == self.clip_len, f"Expected {self.clip_len} frames, got {len(frames)}"
        
        # Resize frames to expected size (default 224x224 for PanEcho, but can be customized)
        target_size = 224  # PanEcho expects 224x224 input
        processed_frames = []
        for frame in frames:
            # First ensure the frame is square by center cropping if needed
            h, w = frame.shape[:2]
            if h != w:
                # Center crop to square
                min_dim = min(h, w)
                start_h = (h - min_dim) // 2
                start_w = (w - min_dim) // 2
                frame = frame[start_h:start_h + min_dim, start_w:start_w + min_dim]
            
            # Resize to target size
            frame = cv2.resize(frame, (target_size, target_size))
            processed_frames.append(frame)
        
        # Stack frames: [frames, height, width, channels]
        frames_array = np.stack(processed_frames)
        
        # Convert to tensor and normalize to [0, 1]
        frames_tensor = torch.from_numpy(frames_array).float() / 255.0
        
        # Normalize with ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
        frames_tensor = (frames_tensor - mean) / std
        
        # Reshape to PanEcho format: [batch, channels, frames, height, width]
        frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)
        
        return frames_tensor.to(self.device)

class VideoQualityFilter:
    """Filter videos based on feature entropy comparison"""
    def __init__(self, feature_extractor: PanEchoFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.original_metrics: List[VideoMetrics] = []
        self.generated_metrics: List[VideoMetrics] = []
        
    def process_videos(self, video_paths: List[str], is_original: bool = True, 
                      use_multiple_clips: bool = False, stride: int = 16) -> List[VideoMetrics]:
        """
        Process a list of videos and extract their features
        
        Args:
            video_paths: List of video file paths
            is_original: Whether these are original training videos
            use_multiple_clips: Whether to extract features from multiple clips per video
            stride: Frame stride when using multiple clips
        """
        metrics = []
        clip_stride = stride if use_multiple_clips else None
        
        for video_path in tqdm(video_paths, desc=f"Processing {'original' if is_original else 'generated'} videos"):
            try:
                features = self.feature_extractor.extract_features(video_path, stride=clip_stride)
                entropy_value = self._calculate_entropy(features)
                
                metric = VideoMetrics(
                    path=video_path,
                    features=features,
                    entropy_value=entropy_value
                )
                metrics.append(metric)
                
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                
        if is_original:
            self.original_metrics = metrics
        else:
            self.generated_metrics = metrics
            
        return metrics
    
    def _calculate_entropy(self, features: np.ndarray) -> float:
        """Calculate entropy of feature vector"""
        # Handle potential NaN or infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Normalize features to probability distribution
        features_pos = features - features.min() + 1e-10
        features_prob = features_pos / features_pos.sum()
        
        # Calculate Shannon entropy
        return entropy(features_prob)
    
    def filter_by_entropy_threshold(self, threshold_percentile: float = 25) -> Tuple[List[VideoMetrics], List[VideoMetrics]]:
        """
        Filter generated videos based on entropy comparison with original videos.
        Videos with entropy too far from original distribution are marked as bad.
        """
        if not self.original_metrics or not self.generated_metrics:
            raise ValueError("Process both original and generated videos first")
        
        # Calculate entropy statistics from original videos
        original_entropies = [m.entropy_value for m in self.original_metrics]
        mean_entropy = np.mean(original_entropies)
        std_entropy = np.std(original_entropies)
        
        # Define threshold based on percentile
        lower_threshold = np.percentile(original_entropies, threshold_percentile)
        upper_threshold = np.percentile(original_entropies, 100 - threshold_percentile)
        
        # Alternative: Use standard deviation based thresholds
        # lower_threshold = mean_entropy - 2 * std_entropy
        # upper_threshold = mean_entropy + 2 * std_entropy
        
        # Filter generated videos
        good_videos = []
        bad_videos = []
        
        for metric in self.generated_metrics:
            if lower_threshold <= metric.entropy_value <= upper_threshold:
                metric.is_good = True
                good_videos.append(metric)
            else:
                metric.is_good = False
                bad_videos.append(metric)
        
        logger.info(f"Original videos entropy: mean={mean_entropy:.4f}, std={std_entropy:.4f}")
        logger.info(f"Entropy thresholds: [{lower_threshold:.4f}, {upper_threshold:.4f}]")
        logger.info(f"Good videos: {len(good_videos)}, Bad videos: {len(bad_videos)}")
        
        return good_videos, bad_videos
    
    def filter_by_feature_distance(self, distance_threshold: float = 2.0) -> Tuple[List[VideoMetrics], List[VideoMetrics]]:
        """
        Alternative filtering method using feature distance to original videos
        """
        if not self.original_metrics or not self.generated_metrics:
            raise ValueError("Process both original and generated videos first")
        
        # Calculate mean feature vector of original videos
        original_features = np.stack([m.features for m in self.original_metrics])
        mean_features = np.mean(original_features, axis=0)
        std_features = np.std(original_features, axis=0)
        
        good_videos = []
        bad_videos = []
        
        for metric in self.generated_metrics:
            # Calculate normalized L2 distance
            distance = np.linalg.norm((metric.features - mean_features) / (std_features + 1e-10))
            
            if distance <= distance_threshold:
                metric.is_good = True
                good_videos.append(metric)
            else:
                metric.is_good = False
                bad_videos.append(metric)
        
        logger.info(f"Feature distance threshold: {distance_threshold}")
        logger.info(f"Good videos: {len(good_videos)}, Bad videos: {len(bad_videos)}")
        
        return good_videos, bad_videos
    
    def visualize_results(self, save_path: str = "panecho_filtering_results.png"):
        """Visualize entropy distributions and feature space"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Entropy distribution
        original_entropies = [m.entropy_value for m in self.original_metrics]
        generated_entropies = [m.entropy_value for m in self.generated_metrics]
        
        ax1.hist(original_entropies, bins=30, alpha=0.5, label='Original', color='blue')
        ax1.hist(generated_entropies, bins=30, alpha=0.5, label='Generated', color='orange')
        ax1.set_xlabel('Entropy')
        ax1.set_ylabel('Count')
        ax1.set_title('Entropy Distribution')
        ax1.legend()
        
        # 2. Quality classification
        good_entropies = [m.entropy_value for m in self.generated_metrics if m.is_good]
        bad_entropies = [m.entropy_value for m in self.generated_metrics if not m.is_good]
        
        if good_entropies and bad_entropies:
            ax2.hist(good_entropies, bins=20, alpha=0.5, label='Good', color='green')
            ax2.hist(bad_entropies, bins=20, alpha=0.5, label='Bad', color='red')
            ax2.set_xlabel('Entropy')
            ax2.set_ylabel('Count')
            ax2.set_title('Quality Classification')
            ax2.legend()
        
        # 3. Entropy box plot
        data_to_plot = [original_entropies]
        labels = ['Original']
        if good_entropies:
            data_to_plot.append(good_entropies)
            labels.append('Generated (Good)')
        if bad_entropies:
            data_to_plot.append(bad_entropies)
            labels.append('Generated (Bad)')
        
        ax3.boxplot(data_to_plot, labels=labels)
        ax3.set_ylabel('Entropy')
        ax3.set_title('Entropy Distribution by Category')
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature statistics
        ax4.text(0.1, 0.9, f"Total Original Videos: {len(self.original_metrics)}", transform=ax4.transAxes)
        ax4.text(0.1, 0.8, f"Total Generated Videos: {len(self.generated_metrics)}", transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f"Good Videos: {len(good_entropies)} ({len(good_entropies)/len(self.generated_metrics)*100:.1f}%)", transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f"Bad Videos: {len(bad_entropies)} ({len(bad_entropies)/len(self.generated_metrics)*100:.1f}%)", transform=ax4.transAxes)
        ax4.text(0.1, 0.4, f"Mean Original Entropy: {np.mean(original_entropies):.4f}", transform=ax4.transAxes)
        ax4.text(0.1, 0.3, f"Std Original Entropy: {np.std(original_entropies):.4f}", transform=ax4.transAxes)
        ax4.axis('off')
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Visualization saved to {save_path}")
    
    def save_results(self, output_path: str = "panecho_filtering_results.json"):
        """Save filtering results to JSON file"""
        results = {
            "model": "PanEcho",
            "clip_length": self.feature_extractor.clip_len,
            "video_size": self.feature_extractor.video_size,
            "statistics": {
                "total_original": len(self.original_metrics),
                "total_generated": len(self.generated_metrics),
                "good_videos": len([m for m in self.generated_metrics if m.is_good]),
                "bad_videos": len([m for m in self.generated_metrics if not m.is_good])
            },
            "good_videos": [m.path for m in self.generated_metrics if m.is_good],
            "bad_videos": [m.path for m in self.generated_metrics if not m.is_good],
            "entropy_values": {
                "original": {m.path: float(m.entropy_value) for m in self.original_metrics},
                "generated": {m.path: float(m.entropy_value) for m in self.generated_metrics}
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Filter generated videos based on PanEcho feature entropy comparison"
    )
    
    # Required arguments
    parser.add_argument(
        "--original_dir",
        type=str,
        required=True,
        help="Directory containing original training videos"
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        required=True,
        help="Directory containing generated videos to filter"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./filtering_results",
        help="Directory to save results (default: ./filtering_results)"
    )
    parser.add_argument(
        "--video_size",
        type=int,
        default=112,
        help="Input video resolution (assumes square videos, default: 112)"
    )
    parser.add_argument(
        "--clip_len",
        type=int,
        default=32,
        help="Number of frames per clip for PanEcho (default: 32)"
    )
    parser.add_argument(
        "--threshold_percentile",
        type=float,
        default=25,
        help="Percentile threshold for filtering (default: 25, filters videos outside 25-75 percentile)"
    )
    parser.add_argument(
        "--filter_method",
        type=str,
        choices=["entropy", "distance"],
        default="entropy",
        help="Filtering method: entropy or feature distance (default: entropy)"
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=2.0,
        help="Distance threshold for feature distance filtering (default: 2.0)"
    )
    parser.add_argument(
        "--use_multiple_clips",
        action="store_true",
        help="Extract features from multiple clips per video"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Frame stride when using multiple clips (default: 16)"
    )
    parser.add_argument(
        "--video_extension",
        type=str,
        default="mp4",
        help="Video file extension (default: mp4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: cuda or cpu (default: cuda)"
    )
    parser.add_argument(
        "--copy_good_videos",
        action="store_true",
        help="Copy good videos to a separate directory"
    )
    parser.add_argument(
        "--copy_bad_videos",
        action="store_true",
        help="Copy bad videos to a separate directory"
    )
    
    return parser.parse_args()

def copy_videos_to_directory(videos: List[VideoMetrics], output_dir: str, category: str):
    """Copy videos to a categorized directory"""
    import shutil
    
    category_dir = os.path.join(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    for video in tqdm(videos, desc=f"Copying {category} videos"):
        src_path = video.path
        dst_path = os.path.join(category_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
    
    logger.info(f"Copied {len(videos)} {category} videos to {category_dir}")

def main():
    """Main function to run the video quality filtering pipeline with PanEcho"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(args.output_dir, "filtering_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log configuration
    logger.info("Starting video filtering with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Initialize PanEcho feature extractor
    extractor = PanEchoFeatureExtractor(
        clip_len=args.clip_len,
        video_size=args.video_size,
        device=args.device
    )
    
    # Initialize quality filter
    filter = VideoQualityFilter(extractor)
    
    # Get video paths
    original_videos = list(Path(args.original_dir).glob(f"*.{args.video_extension}"))
    generated_videos = list(Path(args.generated_dir).glob(f"*.{args.video_extension}"))
    
    if not original_videos:
        logger.error(f"No {args.video_extension} videos found in {args.original_dir}")
        return
    
    if not generated_videos:
        logger.error(f"No {args.video_extension} videos found in {args.generated_dir}")
        return
    
    logger.info(f"Found {len(original_videos)} original videos and {len(generated_videos)} generated videos")
    
    # Process videos
    logger.info("Processing original videos...")
    filter.process_videos(
        [str(p) for p in original_videos], 
        is_original=True, 
        use_multiple_clips=args.use_multiple_clips,
        stride=args.stride
    )
    
    logger.info("Processing generated videos...")
    filter.process_videos(
        [str(p) for p in generated_videos], 
        is_original=False, 
        use_multiple_clips=args.use_multiple_clips,
        stride=args.stride
    )
    
    # Filter videos based on selected method
    if args.filter_method == "entropy":
        good_videos, bad_videos = filter.filter_by_entropy_threshold(args.threshold_percentile)
    else:  # distance
        good_videos, bad_videos = filter.filter_by_feature_distance(args.distance_threshold)
    
    # Visualize results
    viz_path = os.path.join(args.output_dir, "filtering_visualization.png")
    filter.visualize_results(viz_path)
    
    # Save results
    results_path = os.path.join(args.output_dir, "filtering_results.json")
    filter.save_results(results_path)
    
    # Copy videos if requested
    if args.copy_good_videos:
        copy_videos_to_directory(good_videos, args.output_dir, "good_videos")
    
    if args.copy_bad_videos:
        copy_videos_to_directory(bad_videos, args.output_dir, "bad_videos")
    
    # Save lists of good and bad videos
    with open(os.path.join(args.output_dir, "good_videos.txt"), "w") as f:
        for video in good_videos:
            f.write(f"{video.path}\n")
    
    with open(os.path.join(args.output_dir, "bad_videos.txt"), "w") as f:
        for video in bad_videos:
            f.write(f"{video.path}\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Filtering Complete!")
    print(f"{'='*60}")
    print(f"Original videos: {len(original_videos)}")
    print(f"Generated videos: {len(generated_videos)}")
    print(f"Good videos: {len(good_videos)} ({len(good_videos)/len(generated_videos)*100:.1f}%)")
    print(f"Bad videos: {len(bad_videos)} ({len(bad_videos)/len(generated_videos)*100:.1f}%)")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Print example good and bad videos
    if good_videos:
        print(f"Example good videos:")
        for video in good_videos[:5]:
            print(f"  ✓ {os.path.basename(video.path)} (entropy: {video.entropy_value:.4f})")
    
    if bad_videos:
        print(f"\nExample bad videos:")
        for video in bad_videos[:5]:
            print(f"  ✗ {os.path.basename(video.path)} (entropy: {video.entropy_value:.4f})")

if __name__ == "__main__":
    main()
