"""
PanEcho-based Hybrid Video Quality Filter

This script filters generated videos using multiple quality metrics:
1. Per-video entropy (internal complexity)
2. Distribution matching (KL divergence / GMM likelihood)
3. Feature space outlier detection (Mahalanobis distance)

Usage:
    python hybrid_video_filter.py --original_dir /path/to/original --generated_dir /path/to/generated

Example:
    python hybrid_video_filter.py 
        --original_dir /nfs/usrhome/khmuhammad/EchoPath/datasets/CardiacASD/mp4 
        --generated_dir /nfs/scratch/EchoPath/samples/lvdm_asd_triplets/mp4
        --output_dir /nfs/usrhome/khmuhammad/EchoPath/asd_filter_results 
        --filter_mode hybrid
        --copy_good_videos
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import cv2
from typing import List, Tuple, Dict, Optional, Set
from scipy.stats import entropy, multivariate_normal
from scipy.spatial.distance import mahalanobis
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import logging
import argparse
import os
import shutil
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoMetrics:
    """Store comprehensive metrics for a video"""
    path: str
    features: np.ndarray
    entropy_value: float
    distribution_score: float = 0.0
    feature_distance: float = 0.0
    is_good_entropy: bool = True
    is_good_distribution: bool = True
    is_good_distance: bool = True
    is_good_hybrid: bool = True

class PanEchoFeatureExtractor:
    """Video feature extraction using PanEcho model"""
    def __init__(self, clip_len: int = 32, video_size: int = 112, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.clip_len = clip_len
        self.video_size = video_size
        self.feature_output = None
        self.hook = None
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
                
                force_reload = attempt > 0
                
                if force_reload:
                    logger.info("Force reloading model due to previous failure")
                
                model = torch.hub.load(
                    'CarDS-Yale/PanEcho', 
                    'PanEcho', 
                    force_reload=force_reload,
                    clip_len=self.clip_len,
                    trust_repo=True
                )
                
                if model is None:
                    raise RuntimeError("Model loaded but returned None")
                
                # Register hook on the encoder to capture feature extractor output
                self.hook = model.encoder.register_forward_hook(self._hook_fn)
                logger.info("Registered hook on encoder to capture feature extractor output")
                
                # Test model with dummy input
                dummy_input = torch.randn(1, 3, self.clip_len, 224, 224).to(self.device)
                model = model.to(self.device).eval()
                
                with torch.no_grad():
                    try:
                        test_output = model(dummy_input)
                        if test_output is None:
                            raise RuntimeError("Model forward pass returned None")
                        
                        if self.feature_output is None:
                            raise RuntimeError("Failed to capture encoder output via hook")
                        
                        logger.info(f"Model test successful, captured encoder features shape: {self.feature_output.shape}")
                        
                        self.feature_output = None
                            
                    except Exception as e:
                        raise RuntimeError(f"Model forward pass failed: {e}")
                
                logger.info("PanEcho model loaded and verified successfully")
                return model
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                
                if self.hook is not None:
                    self.hook.remove()
                    self.hook = None
                
                if attempt < max_retries - 1:
                    logger.info("Clearing cache and retrying...")
                    self._clear_torch_hub_cache()
                    
                    try:
                        hub_dir = torch.hub.get_dir()
                        panecho_dir = os.path.join(hub_dir, "CarDS-Yale_PanEcho_main")
                        if os.path.exists(panecho_dir):
                            logger.info(f"Removing potentially corrupted PanEcho directory: {panecho_dir}")
                            shutil.rmtree(panecho_dir)
                    except Exception as cleanup_error:
                        logger.warning(f"Could not clean up PanEcho directory: {cleanup_error}")
                    
                    import time
                    time.sleep(2)
                else:
                    logger.error("All attempts to load PanEcho model failed")
                    raise RuntimeError(f"Failed to load PanEcho model after {max_retries} attempts. Original error: {e}")
    
    def extract_features(self, video_path: str, stride: Optional[int] = None) -> np.ndarray:
        """Extract features from a video using PanEcho model"""
        all_frames = self._load_all_video_frames(video_path)
        
        if len(all_frames) < self.clip_len:
            logger.warning(f"Video {video_path} has fewer than {self.clip_len} frames. Padding...")
            all_frames = self._pad_frames(all_frames, self.clip_len)
        
        if stride is None:
            start_idx = max(0, (len(all_frames) - self.clip_len) // 2)
            frames = all_frames[start_idx:start_idx + self.clip_len]
            features = self._extract_clip_features(frames)
        else:
            features_list = []
            for start_idx in range(0, len(all_frames) - self.clip_len + 1, stride):
                frames = all_frames[start_idx:start_idx + self.clip_len]
                clip_features = self._extract_clip_features(frames)
                features_list.append(clip_features)
            
            features = np.mean(features_list, axis=0)
        
        return features.flatten()
    
    def _extract_clip_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract features from a single clip of frames"""
        frames_tensor = self._preprocess_frames(frames)
        
        self.feature_output = None
        
        with torch.no_grad():
            output = self.model(frames_tensor)
            
        if self.feature_output is None:
            raise RuntimeError("Failed to capture encoder features via hook")
        
        features = self.feature_output
        
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
        assert len(frames) == self.clip_len, f"Expected {self.clip_len} frames, got {len(frames)}"
        
        target_size = 224
        processed_frames = []
        for frame in frames:
            h, w = frame.shape[:2]
            if h != w:
                min_dim = min(h, w)
                start_h = (h - min_dim) // 2
                start_w = (w - min_dim) // 2
                frame = frame[start_h:start_h + min_dim, start_w:start_w + min_dim]
            
            frame = cv2.resize(frame, (target_size, target_size))
            processed_frames.append(frame)
        
        frames_array = np.stack(processed_frames)
        frames_tensor = torch.from_numpy(frames_array).float() / 255.0
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
        frames_tensor = (frames_tensor - mean) / std
        
        frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)
        
        return frames_tensor.to(self.device)

class HybridVideoQualityFilter:
    """Filter videos using multiple quality metrics"""
    def __init__(self, feature_extractor: PanEchoFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.original_metrics: List[VideoMetrics] = []
        self.generated_metrics: List[VideoMetrics] = []
        
        # Models for distribution matching
        self.gmm_model = None
        self.pca_model = None
        self.scaler = StandardScaler()
        self.covariance_model = EmpiricalCovariance()
        
    def process_videos(self, video_paths: List[str], is_original: bool = True, 
                      use_multiple_clips: bool = False, stride: int = 16) -> List[VideoMetrics]:
        """Process videos and extract features"""
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
            self._fit_distribution_models()
        else:
            self.generated_metrics = metrics
            self._calculate_distribution_scores()
            
        return metrics
    
    def _calculate_entropy(self, features: np.ndarray) -> float:
        """Calculate entropy of feature vector"""
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        features_pos = features - features.min() + 1e-10
        features_prob = features_pos / features_pos.sum()
        return entropy(features_prob)
    
    def _fit_distribution_models(self):
        """Fit distribution models on original videos"""
        if not self.original_metrics:
            return
            
        logger.info("Fitting distribution models on original videos...")
        
        # Stack all original features
        original_features = np.stack([m.features for m in self.original_metrics])
        
        # Fit StandardScaler
        self.scaler.fit(original_features)
        scaled_features = self.scaler.transform(original_features)
        
        # Fit PCA for dimensionality reduction
        n_components = min(50, scaled_features.shape[0] - 1, scaled_features.shape[1])
        self.pca_model = PCA(n_components=n_components, whiten=True)
        pca_features = self.pca_model.fit_transform(scaled_features)
        
        # Fit GMM on PCA features
        n_gmm_components = min(5, len(self.original_metrics) // 10 + 1)
        self.gmm_model = GaussianMixture(
            n_components=n_gmm_components,
            covariance_type='full',
            random_state=42
        )
        self.gmm_model.fit(pca_features)
        
        # Fit covariance model for Mahalanobis distance
        self.covariance_model.fit(scaled_features)
        
        logger.info(f"Fitted models: PCA({n_components}), GMM({n_gmm_components})")
    
    def _calculate_distribution_scores(self):
        """Calculate distribution scores for generated videos"""
        if not self.generated_metrics or self.gmm_model is None:
            return
            
        for metric in self.generated_metrics:
            # Transform features
            scaled_features = self.scaler.transform(metric.features.reshape(1, -1))
            pca_features = self.pca_model.transform(scaled_features)
            
            # GMM log-likelihood
            metric.distribution_score = self.gmm_model.score_samples(pca_features)[0]
            
            # Mahalanobis distance
            try:
                metric.feature_distance = self.covariance_model.mahalanobis(scaled_features[0])
            except:
                # Fallback to Euclidean distance if Mahalanobis fails
                mean_features = self.scaler.mean_
                metric.feature_distance = np.linalg.norm(metric.features - mean_features)
    
    def filter_by_entropy_threshold(self, threshold_percentile: float = 25) -> Tuple[List[VideoMetrics], List[VideoMetrics]]:
        """Filter by per-video entropy"""
        if not self.original_metrics or not self.generated_metrics:
            raise ValueError("Process both original and generated videos first")
        
        original_entropies = [m.entropy_value for m in self.original_metrics]
        lower_threshold = np.percentile(original_entropies, threshold_percentile)
        upper_threshold = np.percentile(original_entropies, 100 - threshold_percentile)
        
        good_videos = []
        bad_videos = []
        
        for metric in self.generated_metrics:
            if lower_threshold <= metric.entropy_value <= upper_threshold:
                metric.is_good_entropy = True
                good_videos.append(metric)
            else:
                metric.is_good_entropy = False
                bad_videos.append(metric)
        
        logger.info(f"Entropy filter: Good={len(good_videos)}, Bad={len(bad_videos)}")
        return good_videos, bad_videos
    
    def filter_by_distribution(self, threshold_percentile: float = 10) -> Tuple[List[VideoMetrics], List[VideoMetrics]]:
        """Filter by distribution matching (GMM likelihood)"""
        if not self.generated_metrics or self.gmm_model is None:
            raise ValueError("Process videos and fit models first")
        
        # Calculate threshold from original videos
        original_features = np.stack([m.features for m in self.original_metrics])
        scaled_features = self.scaler.transform(original_features)
        pca_features = self.pca_model.transform(scaled_features)
        original_scores = self.gmm_model.score_samples(pca_features)
        
        threshold = np.percentile(original_scores, threshold_percentile)
        
        good_videos = []
        bad_videos = []
        
        for metric in self.generated_metrics:
            if metric.distribution_score >= threshold:
                metric.is_good_distribution = True
                good_videos.append(metric)
            else:
                metric.is_good_distribution = False
                bad_videos.append(metric)
        
        logger.info(f"Distribution filter: Good={len(good_videos)}, Bad={len(bad_videos)}")
        return good_videos, bad_videos
    
    def filter_by_feature_distance(self, threshold_percentile: float = 95) -> Tuple[List[VideoMetrics], List[VideoMetrics]]:
        """Filter by feature space distance (Mahalanobis)"""
        if not self.generated_metrics:
            raise ValueError("Process generated videos first")
        
        # Calculate threshold from original videos
        original_features = np.stack([m.features for m in self.original_metrics])
        scaled_features = self.scaler.transform(original_features)
        original_distances = []
        
        for feat in scaled_features:
            try:
                dist = self.covariance_model.mahalanobis(feat)
                original_distances.append(dist)
            except:
                pass
        
        if not original_distances:
            # Fallback to using generated video distances
            threshold = np.percentile([m.feature_distance for m in self.generated_metrics], threshold_percentile)
        else:
            threshold = np.percentile(original_distances, threshold_percentile)
        
        good_videos = []
        bad_videos = []
        
        for metric in self.generated_metrics:
            if metric.feature_distance <= threshold:
                metric.is_good_distance = True
                good_videos.append(metric)
            else:
                metric.is_good_distance = False
                bad_videos.append(metric)
        
        logger.info(f"Distance filter: Good={len(good_videos)}, Bad={len(bad_videos)}")
        return good_videos, bad_videos
    
    def apply_hybrid_filter(self, strategy: str = "all") -> Tuple[List[VideoMetrics], List[VideoMetrics]]:
        """
        Apply hybrid filtering with different strategies
        
        Args:
            strategy: 'all' (must pass all), 'any' (pass any), 'majority' (pass 2/3)
        """
        good_videos = []
        bad_videos = []
        
        for metric in self.generated_metrics:
            passed_filters = sum([
                metric.is_good_entropy,
                metric.is_good_distribution,
                metric.is_good_distance
            ])
            
            if strategy == "all":
                metric.is_good_hybrid = passed_filters == 3
            elif strategy == "any":
                metric.is_good_hybrid = passed_filters >= 1
            elif strategy == "majority":
                metric.is_good_hybrid = passed_filters >= 2
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            if metric.is_good_hybrid:
                good_videos.append(metric)
            else:
                bad_videos.append(metric)
        
        logger.info(f"Hybrid filter ({strategy}): Good={len(good_videos)}, Bad={len(bad_videos)}")
        return good_videos, bad_videos
    
    def visualize_comprehensive_results(self, save_path: str = "hybrid_filtering_results.png"):
        """Create comprehensive visualization of all filtering methods"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Color scheme
        colors = {
            'original': '#3498db',
            'good': '#2ecc71', 
            'bad': '#e74c3c',
            'generated': '#f39c12'
        }
        
        # 1. Entropy Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        original_entropies = [m.entropy_value for m in self.original_metrics]
        generated_entropies = [m.entropy_value for m in self.generated_metrics]
        
        ax1.hist(original_entropies, bins=20, alpha=0.7, label='Original', color=colors['original'], density=True)
        ax1.hist(generated_entropies, bins=20, alpha=0.7, label='Generated', color=colors['generated'], density=True)
        ax1.set_xlabel('Entropy')
        ax1.set_ylabel('Density')
        ax1.set_title('Entropy Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution Scores
        ax2 = fig.add_subplot(gs[0, 1])
        if self.gmm_model is not None:
            generated_scores = [m.distribution_score for m in self.generated_metrics]
            good_scores = [m.distribution_score for m in self.generated_metrics if m.is_good_distribution]
            bad_scores = [m.distribution_score for m in self.generated_metrics if not m.is_good_distribution]
            
            if good_scores:
                ax2.hist(good_scores, bins=15, alpha=0.7, label='Good', color=colors['good'], density=True)
            if bad_scores:
                ax2.hist(bad_scores, bins=15, alpha=0.7, label='Bad', color=colors['bad'], density=True)
            ax2.set_xlabel('GMM Log-Likelihood')
            ax2.set_ylabel('Density')
            ax2.set_title('Distribution Matching Scores')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Feature Distances
        ax3 = fig.add_subplot(gs[0, 2])
        generated_distances = [m.feature_distance for m in self.generated_metrics]
        good_distances = [m.feature_distance for m in self.generated_metrics if m.is_good_distance]
        bad_distances = [m.feature_distance for m in self.generated_metrics if not m.is_good_distance]
        
        if good_distances:
            ax3.hist(good_distances, bins=15, alpha=0.7, label='Good', color=colors['good'], density=True)
        if bad_distances:
            ax3.hist(bad_distances, bins=15, alpha=0.7, label='Bad', color=colors['bad'], density=True)
        ax3.set_xlabel('Mahalanobis Distance')
        ax3.set_ylabel('Density')
        ax3.set_title('Feature Space Distances')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Filter Comparison (Venn-like)
        ax4 = fig.add_subplot(gs[1, :])
        
        # Count videos passing each filter
        entropy_pass = set([m.path for m in self.generated_metrics if m.is_good_entropy])
        dist_pass = set([m.path for m in self.generated_metrics if m.is_good_distribution])
        feat_pass = set([m.path for m in self.generated_metrics if m.is_good_distance])
        
        # Calculate intersections
        all_three = len(entropy_pass & dist_pass & feat_pass)
        entropy_dist = len((entropy_pass & dist_pass) - feat_pass)
        entropy_feat = len((entropy_pass & feat_pass) - dist_pass)
        dist_feat = len((dist_pass & feat_pass) - entropy_pass)
        only_entropy = len(entropy_pass - dist_pass - feat_pass)
        only_dist = len(dist_pass - entropy_pass - feat_pass)
        only_feat = len(feat_pass - entropy_pass - dist_pass)
        none = len(self.generated_metrics) - len(entropy_pass | dist_pass | feat_pass)
        
        # Create bar chart
        categories = ['All 3', 'Entropy+Dist', 'Entropy+Feat', 'Dist+Feat', 
                     'Only Entropy', 'Only Dist', 'Only Feat', 'None']
        values = [all_three, entropy_dist, entropy_feat, dist_feat, 
                 only_entropy, only_dist, only_feat, none]
        
        bars = ax4.bar(categories, values, color=['#2ecc71' if i == 0 else '#95a5a6' for i in range(len(categories))])
        ax4.set_ylabel('Number of Videos')
        ax4.set_title('Filter Overlap Analysis')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(val), ha='center', va='bottom')
        
        # 5. 2D PCA Projection
        ax5 = fig.add_subplot(gs[2, 0:2])
        if self.pca_model is not None and self.pca_model.n_components >= 2:
            # Project all videos to 2D
            all_features = []
            all_labels = []
            all_colors = []
            
            # Original videos
            for m in self.original_metrics:
                scaled = self.scaler.transform(m.features.reshape(1, -1))
                all_features.append(scaled[0])
                all_labels.append('Original')
                all_colors.append(colors['original'])
            
            # Generated videos
            for m in self.generated_metrics:
                scaled = self.scaler.transform(m.features.reshape(1, -1))
                all_features.append(scaled[0])
                if m.is_good_hybrid:
                    all_labels.append('Good')
                    all_colors.append(colors['good'])
                else:
                    all_labels.append('Bad')
                    all_colors.append(colors['bad'])
            
            # PCA projection
            pca_2d = PCA(n_components=2)
            features_2d = pca_2d.fit_transform(np.array(all_features))
            
            # Plot
            for label, color in [('Original', colors['original']), 
                               ('Good', colors['good']), 
                               ('Bad', colors['bad'])]:
                mask = np.array(all_labels) == label
                if np.any(mask):
                    ax5.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                              c=color, label=label, alpha=0.6, s=50)
            
            ax5.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
            ax5.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
            ax5.set_title('Feature Space Visualization (2D PCA)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Summary Statistics
        ax6 = fig.add_subplot(gs[2:, 2])
        ax6.axis('off')
        
        summary_text = f"""Summary Statistics
{'='*30}

Total Videos:
  • Original: {len(self.original_metrics)}
  • Generated: {len(self.generated_metrics)}

Individual Filters:
  • Entropy: {len([m for m in self.generated_metrics if m.is_good_entropy])} pass ({len([m for m in self.generated_metrics if m.is_good_entropy])/len(self.generated_metrics)*100:.1f}%)
  • Distribution: {len([m for m in self.generated_metrics if m.is_good_distribution])} pass ({len([m for m in self.generated_metrics if m.is_good_distribution])/len(self.generated_metrics)*100:.1f}%)
  • Distance: {len([m for m in self.generated_metrics if m.is_good_distance])} pass ({len([m for m in self.generated_metrics if m.is_good_distance])/len(self.generated_metrics)*100:.1f}%)

Hybrid Filter Results:
  • Good: {len([m for m in self.generated_metrics if m.is_good_hybrid])} ({len([m for m in self.generated_metrics if m.is_good_hybrid])/len(self.generated_metrics)*100:.1f}%)
  • Bad: {len([m for m in self.generated_metrics if not m.is_good_hybrid])} ({len([m for m in self.generated_metrics if not m.is_good_hybrid])/len(self.generated_metrics)*100:.1f}%)

Quality Metrics:
  • Mean Entropy (Original): {np.mean(original_entropies):.4f}
  • Std Entropy (Original): {np.std(original_entropies):.4f}
  • Mean Entropy (Generated): {np.mean(generated_entropies):.4f}
  • Std Entropy (Generated): {np.std(generated_entropies):.4f}
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 7. Box plots comparison
        ax7 = fig.add_subplot(gs[3, :2])
        
        data_for_box = []
        labels_for_box = []
        
        # Entropy comparison
        data_for_box.extend([
            [m.entropy_value for m in self.original_metrics],
            [m.entropy_value for m in self.generated_metrics if m.is_good_hybrid],
            [m.entropy_value for m in self.generated_metrics if not m.is_good_hybrid]
        ])
        labels_for_box.extend(['Original', 'Good', 'Bad'])
        
        bp = ax7.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch, color in zip(bp['boxes'], [colors['original'], colors['good'], colors['bad']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax7.set_ylabel('Entropy Value')
        ax7.set_title('Entropy Comparison by Category')
        ax7.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Hybrid Video Quality Filtering Results', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive visualization saved to {save_path}")
    
    def save_results(self, output_path: str = "hybrid_filtering_results.json"):
        """Save comprehensive filtering results"""
        results = {
            "model": "PanEcho",
            "clip_length": self.feature_extractor.clip_len,
            "video_size": self.feature_extractor.video_size,
            "filter_methods": ["entropy", "distribution", "feature_distance"],
            "statistics": {
                "total_original": len(self.original_metrics),
                "total_generated": len(self.generated_metrics),
                "entropy_pass": len([m for m in self.generated_metrics if m.is_good_entropy]),
                "distribution_pass": len([m for m in self.generated_metrics if m.is_good_distribution]),
                "distance_pass": len([m for m in self.generated_metrics if m.is_good_distance]),
                "hybrid_pass": len([m for m in self.generated_metrics if m.is_good_hybrid]),
                "hybrid_fail": len([m for m in self.generated_metrics if not m.is_good_hybrid])
            },
            "good_videos": [m.path for m in self.generated_metrics if m.is_good_hybrid],
            "bad_videos": [m.path for m in self.generated_metrics if not m.is_good_hybrid],
            "detailed_metrics": {
                m.path: {
                    "entropy": float(m.entropy_value),
                    "distribution_score": float(m.distribution_score),
                    "feature_distance": float(m.feature_distance),
                    "passed_entropy": m.is_good_entropy,
                    "passed_distribution": m.is_good_distribution,
                    "passed_distance": m.is_good_distance,
                    "passed_hybrid": m.is_good_hybrid
                }
                for m in self.generated_metrics
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Hybrid video quality filtering using PanEcho features"
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
    
    # Filtering thresholds
    parser.add_argument(
        "--entropy_percentile",
        type=float,
        default=25,
        help="Percentile threshold for entropy filtering (default: 25)"
    )
    parser.add_argument(
        "--distribution_percentile",
        type=float,
        default=10,
        help="Percentile threshold for distribution filtering (default: 10)"
    )
    parser.add_argument(
        "--distance_percentile",
        type=float,
        default=95,
        help="Percentile threshold for distance filtering (default: 95)"
    )
    
    # Filtering mode
    parser.add_argument(
        "--filter_mode",
        type=str,
        choices=["entropy", "distribution", "distance", "hybrid"],
        default="hybrid",
        help="Filtering mode (default: hybrid)"
    )
    parser.add_argument(
        "--hybrid_strategy",
        type=str,
        choices=["all", "any", "majority"],
        default="majority",
        help="Hybrid filtering strategy: all (must pass all), any (pass any), majority (pass 2/3)"
    )
    
    # Feature extraction
    parser.add_argument(
        "--use_multiple_clips",
        action="store_true",
        help="Extract features from multiple clips per video"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Frame stride when using multiple clips (default: 16)"
    )
    
    # Other options
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
    """Main function to run the hybrid video quality filtering pipeline"""
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
    logger.info("Starting hybrid video filtering with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Initialize PanEcho feature extractor
    extractor = PanEchoFeatureExtractor(
        clip_len=args.clip_len,
        video_size=args.video_size,
        device=args.device
    )
    
    # Initialize hybrid quality filter
    filter = HybridVideoQualityFilter(extractor)
    
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
    
    # Apply filters based on mode
    if args.filter_mode == "entropy":
        good_videos, bad_videos = filter.filter_by_entropy_threshold(args.entropy_percentile)
    elif args.filter_mode == "distribution":
        filter.filter_by_entropy_threshold(args.entropy_percentile)  # Need to run all filters for hybrid
        good_videos, bad_videos = filter.filter_by_distribution(args.distribution_percentile)
    elif args.filter_mode == "distance":
        filter.filter_by_entropy_threshold(args.entropy_percentile)  # Need to run all filters for hybrid
        filter.filter_by_distribution(args.distribution_percentile)
        good_videos, bad_videos = filter.filter_by_feature_distance(args.distance_percentile)
    else:  # hybrid
        # Run all individual filters
        filter.filter_by_entropy_threshold(args.entropy_percentile)
        filter.filter_by_distribution(args.distribution_percentile)
        filter.filter_by_feature_distance(args.distance_percentile)
        # Apply hybrid strategy
        good_videos, bad_videos = filter.apply_hybrid_filter(args.hybrid_strategy)
    
    # Visualize results
    viz_path = os.path.join(args.output_dir, "hybrid_filtering_visualization.png")
    filter.visualize_comprehensive_results(viz_path)
    
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
    print(f"Hybrid Filtering Complete!")
    print(f"{'='*60}")
    print(f"Original videos: {len(original_videos)}")
    print(f"Generated videos: {len(generated_videos)}")
    print(f"Filter mode: {args.filter_mode}")
    if args.filter_mode == "hybrid":
        print(f"Hybrid strategy: {args.hybrid_strategy}")
    print(f"Good videos: {len(good_videos)} ({len(good_videos)/len(generated_videos)*100:.1f}%)")
    print(f"Bad videos: {len(bad_videos)} ({len(bad_videos)/len(generated_videos)*100:.1f}%)")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Print filter breakdown for hybrid mode
    if args.filter_mode == "hybrid":
        print("Filter Breakdown:")
        print(f"  • Entropy: {len([m for m in filter.generated_metrics if m.is_good_entropy])} pass")
        print(f"  • Distribution: {len([m for m in filter.generated_metrics if m.is_good_distribution])} pass")
        print(f"  • Distance: {len([m for m in filter.generated_metrics if m.is_good_distance])} pass")
        print(f"  • All three: {len([m for m in filter.generated_metrics if m.is_good_entropy and m.is_good_distribution and m.is_good_distance])}")
        print()
    
    # Print example good and bad videos
    if good_videos:
        print(f"Example good videos:")
        for video in good_videos[:5]:
            print(f"  ✓ {os.path.basename(video.path)}")
            print(f"    - Entropy: {video.entropy_value:.4f} {'✓' if video.is_good_entropy else '✗'}")
            print(f"    - Distribution: {video.distribution_score:.4f} {'✓' if video.is_good_distribution else '✗'}")
            print(f"    - Distance: {video.feature_distance:.4f} {'✓' if video.is_good_distance else '✗'}")
    
    if bad_videos:
        print(f"\nExample bad videos:")
        for video in bad_videos[:5]:
            print(f"  ✗ {os.path.basename(video.path)}")
            print(f"    - Entropy: {video.entropy_value:.4f} {'✓' if video.is_good_entropy else '✗'}")
            print(f"    - Distribution: {video.distribution_score:.4f} {'✓' if video.is_good_distribution else '✗'}")
            print(f"    - Distance: {video.feature_distance:.4f} {'✓' if video.is_good_distance else '✗'}")

if __name__ == "__main__":
    main()
