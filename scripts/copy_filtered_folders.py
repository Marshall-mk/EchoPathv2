"""
Copy Filtered Video Folders Script

This script reads a list of video file paths and copies the corresponding 
folders containing JPEG frames to a new directory.

Usage:
    python copy_filtered_folders.py --video_list good_videos.txt --source_dir /path/to/frames --output_dir /path/to/filtered_frames

Example:
    python copy_filtered_folders.py 
        --video_list /nfs/usrhome/khmuhammad/EchoPath/asd_filter_results/good_videos.txt
        --source_dir /nfs/scratch/EchoPath/samples/lvdm_asd_triplets/jpg
        --output_dir /nfs/scratch/EchoPath/samples/lvdm_asd_triplets_filtered/jpg
        --dry_run
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_folder_name_from_video_path(video_path: str) -> str:
    """
    Extract folder name from video file path.
    
    Args:
        video_path: Path to video file (e.g., /path/to/sample_000001.mp4)
    
    Returns:
        Folder name (e.g., sample_000001)
    """
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(video_path))[0]
    return filename

def read_video_paths(txt_file: str) -> list:
    """
    Read video paths from txt file.
    
    Args:
        txt_file: Path to txt file containing video paths
    
    Returns:
        List of video file paths
    """
    video_paths = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                video_paths.append(line)
    return video_paths

def copy_folders(video_paths: list, source_dir: str, output_dir: str, 
                 dry_run: bool = False) -> tuple:
    """
    Copy folders corresponding to video paths.
    
    Args:
        video_paths: List of video file paths
        source_dir: Source directory containing folders
        output_dir: Destination directory for copied folders
        dry_run: If True, only show what would be copied without actually copying
    
    Returns:
        Tuple of (successful_copies, failed_copies)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    successful_copies = []
    failed_copies = []
    
    for video_path in tqdm(video_paths, desc="Processing folders"):
        # Extract folder name from video path
        folder_name = extract_folder_name_from_video_path(video_path)
        
        # Construct source and destination paths
        source_folder = os.path.join(source_dir, folder_name)
        dest_folder = os.path.join(output_dir, folder_name)
        
        # Check if source folder exists
        if not os.path.exists(source_folder):
            logger.warning(f"Source folder not found: {source_folder}")
            failed_copies.append(folder_name)
            continue
        
        # Check if destination already exists
        if os.path.exists(dest_folder):
            if dry_run:
                logger.info(f"[DRY RUN] Would skip existing folder: {dest_folder}")
            else:
                logger.info(f"Skipping existing folder: {dest_folder}")
            successful_copies.append(folder_name)
            continue
        
        try:
            if dry_run:
                logger.info(f"[DRY RUN] Would copy: {source_folder} -> {dest_folder}")
            else:
                # Copy the entire folder
                shutil.copytree(source_folder, dest_folder)
                logger.info(f"Copied: {source_folder} -> {dest_folder}")
            
            successful_copies.append(folder_name)
            
        except Exception as e:
            logger.error(f"Failed to copy {source_folder}: {e}")
            failed_copies.append(folder_name)
    
    return successful_copies, failed_copies

def verify_folders(video_paths: list, source_dir: str) -> tuple:
    """
    Verify that corresponding folders exist for all video paths.
    
    Args:
        video_paths: List of video file paths
        source_dir: Source directory containing folders
    
    Returns:
        Tuple of (existing_folders, missing_folders)
    """
    existing_folders = []
    missing_folders = []
    
    for video_path in video_paths:
        folder_name = extract_folder_name_from_video_path(video_path)
        source_folder = os.path.join(source_dir, folder_name)
        
        if os.path.exists(source_folder) and os.path.isdir(source_folder):
            existing_folders.append(folder_name)
        else:
            missing_folders.append(folder_name)
    
    return existing_folders, missing_folders

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Copy folders corresponding to filtered video list"
    )
    
    # Required arguments
    parser.add_argument(
        "--video_list",
        type=str,
        required=True,
        help="Path to txt file containing video file paths"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Source directory containing folders to copy"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Destination directory for copied folders"
    )
    
    # Optional arguments
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be copied without actually copying"
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify that folders exist, don't copy"
    )
    parser.add_argument(
        "--create_symlinks",
        action="store_true",
        help="Create symbolic links instead of copying (saves space)"
    )
    
    return parser.parse_args()

def create_symlinks(video_paths: list, source_dir: str, output_dir: str) -> tuple:
    """
    Create symbolic links instead of copying folders.
    
    Args:
        video_paths: List of video file paths
        source_dir: Source directory containing folders
        output_dir: Destination directory for symbolic links
    
    Returns:
        Tuple of (successful_links, failed_links)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    successful_links = []
    failed_links = []
    
    for video_path in tqdm(video_paths, desc="Creating symbolic links"):
        folder_name = extract_folder_name_from_video_path(video_path)
        source_folder = os.path.abspath(os.path.join(source_dir, folder_name))
        dest_link = os.path.join(output_dir, folder_name)
        
        # Check if source folder exists
        if not os.path.exists(source_folder):
            logger.warning(f"Source folder not found: {source_folder}")
            failed_links.append(folder_name)
            continue
        
        # Check if link already exists
        if os.path.exists(dest_link):
            logger.info(f"Skipping existing link: {dest_link}")
            successful_links.append(folder_name)
            continue
        
        try:
            os.symlink(source_folder, dest_link)
            logger.info(f"Created symlink: {source_folder} -> {dest_link}")
            successful_links.append(folder_name)
            
        except Exception as e:
            logger.error(f"Failed to create symlink for {source_folder}: {e}")
            failed_links.append(folder_name)
    
    return successful_links, failed_links

def main():
    """Main function"""
    args = parse_args()
    
    # Read video paths from txt file
    logger.info(f"Reading video paths from: {args.video_list}")
    try:
        video_paths = read_video_paths(args.video_list)
        logger.info(f"Found {len(video_paths)} video paths")
    except Exception as e:
        logger.error(f"Failed to read video list: {e}")
        return
    
    # Extract folder names for logging
    folder_names = [extract_folder_name_from_video_path(path) for path in video_paths]
    logger.info(f"Extracted {len(folder_names)} folder names")
    
    # Verify that source directory exists
    if not os.path.exists(args.source_dir):
        logger.error(f"Source directory does not exist: {args.source_dir}")
        return
    
    # Verify folders exist
    logger.info("Verifying that corresponding folders exist...")
    existing_folders, missing_folders = verify_folders(video_paths, args.source_dir)
    
    logger.info(f"Existing folders: {len(existing_folders)}")
    logger.info(f"Missing folders: {len(missing_folders)}")
    
    if missing_folders:
        logger.warning("Missing folders:")
        for folder in missing_folders[:10]:  # Show first 10
            logger.warning(f"  - {folder}")
        if len(missing_folders) > 10:
            logger.warning(f"  ... and {len(missing_folders) - 10} more")
    
    if args.verify_only:
        logger.info("Verification complete. Exiting (--verify_only specified).")
        return
    
    # Copy or link folders
    if existing_folders:
        if args.create_symlinks:
            logger.info("Creating symbolic links...")
            successful, failed = create_symlinks(video_paths, args.source_dir, args.output_dir)
            operation = "linked"
        else:
            logger.info("Copying folders...")
            successful, failed = copy_folders(video_paths, args.source_dir, args.output_dir, args.dry_run)
            operation = "copied"
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Operation Complete!")
        print(f"{'='*60}")
        print(f"Total folders to process: {len(existing_folders)}")
        print(f"Successfully {operation}: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Missing from source: {len(missing_folders)}")
        
        if not args.dry_run:
            print(f"Output directory: {args.output_dir}")
        
        print(f"{'='*60}\n")
        
        if failed:
            logger.error("Failed folders:")
            for folder in failed:
                logger.error(f"  - {folder}")
    
    else:
        logger.error("No existing folders found to copy!")

if __name__ == "__main__":
    main()
