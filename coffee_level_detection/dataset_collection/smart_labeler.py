"""
Smart Coffee Level Annotation Tool with CNN-guided prioritization.
Prioritizes images based on CNN confidence for levels 2-10 to address class imbalance.
"""
import cv2
import json
import os
import random
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import hashlib
from datetime import datetime
from coffee_level_detection.dataset_collection.models import CoffeeLevelSelector
from tqdm import tqdm
from collections import defaultdict


def load_cnn_predictions(predictions_file='inference_coffee_level_annotations.json'):
    """Load CNN predictions with confidence scores."""
    if not os.path.exists(predictions_file):
        print(f"Warning: CNN predictions file {predictions_file} not found!")
        return {}
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = {}
    for entry in data.get('annotation_data', []):
        filename = entry.get('filename', '')
        coffee_level = entry.get('coffee_level')
        confidence = entry.get('confidence', 0.0)
        
        if filename and coffee_level is not None:
            predictions[filename] = {
                'predicted_level': coffee_level,
                'confidence': confidence
            }
    
    return predictions


def normalize_filename(filename):
    """Normalize filename by removing prefixes and keeping the base name."""
    if 'masked_' in filename:
        return filename.split('masked_')[-1]
    return filename


def calculate_priority_score(pred_level, confidence, target_levels=[2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """
    Calculate priority score for manual labeling.
    Higher score = higher priority for manual annotation.
    
    Args:
        pred_level: Predicted coffee level
        confidence: Model confidence (0-1)
        target_levels: Levels we want to prioritize for annotation
    
    Returns:
        Priority score (higher = more important)
    """
    base_score = 0
    
    # High priority for predicted levels 2-10 (non-empty)
    if pred_level in target_levels:
        base_score += 10
    
    # Bonus for high confidence predictions (we trust these more)
    confidence_bonus = confidence * 5
    
    # Extra bonus for mid-range levels (3-7) which are often underrepresented
    if pred_level in [3, 4, 5, 6, 7]:
        base_score += 3
    
    # Small penalty for very high levels (9, 10) to avoid over-sampling
    if pred_level in [9, 10]:
        base_score -= 1
    
    return base_score + confidence_bonus


def prioritize_images(img_files, predictions, labeled_files):
    """
    Prioritize images for manual annotation based on CNN predictions.
    
    Args:
        img_files: List of available image files
        predictions: CNN predictions dict
        labeled_files: Set of already labeled files
    
    Returns:
        List of (filename, priority_score, pred_info) tuples, sorted by priority
    """
    priorities = []
    
    for filename in img_files:
        if filename in labeled_files:
            continue
            
        # Try to find prediction for this file (handle filename variations)
        pred_info = None
        normalized_name = normalize_filename(filename)
        
        # Look for predictions under various filename formats
        for pred_filename in predictions:
            if (pred_filename == filename or 
                pred_filename == normalized_name or
                normalize_filename(pred_filename) == normalized_name):
                pred_info = predictions[pred_filename]
                break
        
        if pred_info:
            priority = calculate_priority_score(
                pred_info['predicted_level'], 
                pred_info['confidence']
            )
            priorities.append((filename, priority, pred_info))
        else:
            # If no prediction available, give medium priority
            priorities.append((filename, 5.0, {'predicted_level': 'unknown', 'confidence': 0.0}))
    
    # Sort by priority (descending)
    priorities.sort(key=lambda x: x[1], reverse=True)
    return priorities


def get_cache_key(predictions_file, img_files, labeled_files):
    """Generate a cache key based on input data to detect changes."""
    # Include predictions file modification time and size
    pred_stat = os.stat(predictions_file) if os.path.exists(predictions_file) else None
    pred_info = f"{pred_stat.st_mtime}_{pred_stat.st_size}" if pred_stat else "no_predictions"
    
    # Include count of image files and labeled files
    img_info = f"imgs_{len(img_files)}"
    labeled_info = f"labeled_{len(labeled_files)}"
    
    # Create hash from combined info
    combined_info = f"{pred_info}_{img_info}_{labeled_info}"
    return hashlib.md5(combined_info.encode()).hexdigest()


def load_cached_priorities(cache_file):
    """Load cached priorities if they exist."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            return cache_data
        except Exception as e:
            print(f"Warning: Could not load cache file {cache_file}: {e}")
    return None


def save_cached_priorities(cache_file, cache_key, prioritized_images):
    """Save priorities to cache file."""
    try:
        cache_data = {
            'cache_key': cache_key,
            'timestamp': datetime.now().isoformat(),
            'prioritized_images': prioritized_images
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"💾 Saved priority cache to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache file {cache_file}: {e}")


def get_prioritized_images_cached(img_files, predictions, labeled_files, predictions_file, 
                                cache_file='priority_cache.pkl', force_refresh=False):
    """
    Get prioritized images with caching support.
    
    Args:
        img_files: List of available image files
        predictions: CNN predictions dict
        labeled_files: Set of already labeled files
        predictions_file: Path to predictions file for cache key
        cache_file: Cache file path
        force_refresh: Force recalculation even if cache exists
    
    Returns:
        List of prioritized images
    """
    # Generate cache key
    cache_key = get_cache_key(predictions_file, img_files, labeled_files)
    
    # Try to load from cache
    if not force_refresh:
        cached_data = load_cached_priorities(cache_file)
        if cached_data and cached_data.get('cache_key') == cache_key:
            print(f"🚀 Loaded priorities from cache ({len(cached_data['prioritized_images'])} images)")
            print(f"📅 Cache created: {cached_data.get('timestamp', 'unknown')}")
            return cached_data['prioritized_images']
    
    # Cache miss - calculate priorities
    print("🔄 Calculating image priorities (this may take a moment)...")
    prioritized_images = prioritize_images(img_files, predictions, labeled_files)
    
    # Save to cache
    save_cached_priorities(cache_file, cache_key, prioritized_images)
    
    return prioritized_images


def show_priority_summary(prioritized_images, num_show=20):
    """Show summary of prioritized images."""
    print(f"\n=== TOP {num_show} PRIORITY IMAGES FOR MANUAL LABELING ===")
    print(f"{'Rank':<4} {'Filename':<40} {'Pred Level':<10} {'Confidence':<10} {'Priority':<8}")
    print("-" * 80)
    
    level_counts = defaultdict(int)
    for i, (filename, priority, pred_info) in enumerate(prioritized_images[:num_show]):
        pred_level = pred_info['predicted_level']
        confidence = pred_info['confidence']
        level_counts[pred_level] += 1
        
        print(f"{i+1:<4} {filename:<40} {pred_level:<10} {confidence:<10.3f} {priority:<8.1f}")
    
    print(f"\nPredicted level distribution in top {num_show}:")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} images")
    print()


def main():
    """Main function for smart coffee level annotation."""
    # Setup
    IMG_DIR = 'processed_images'
    manual_levels_dir = 'manual_levels'
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Smart Coffee Level Annotator')
    parser.add_argument('--samples', type=int, default=10, 
                       help='Number of images to process')
    parser.add_argument('--predictions', type=str, 
                       default='inference_coffee_level_annotations.json',
                       help='CNN predictions file')
    parser.add_argument('--show-summary', action='store_true',
                       help='Show prioritization summary and exit')
    parser.add_argument('--prioritize', action='store_true', default=True,
                       help='Use CNN-guided prioritization (default: True)')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh of priority cache')
    args = parser.parse_args()
    
    # Load available images
    if not os.path.exists(IMG_DIR):
        print(f"Error: Image directory {IMG_DIR} not found!")
        return
    
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    
    # Load already labeled files
    labeled_files = set()
    if os.path.exists(manual_levels_dir):
        for level_file in os.listdir(manual_levels_dir):
            if level_file.startswith('level_') and level_file.endswith('.json'):
                original_name = level_file[6:-5]  # Remove 'level_' and '.json'
                labeled_files.add(original_name)
    
    # Filter out already labeled files
    unlabeled_files = [f for f in img_files if f not in labeled_files]
    
    print(f"Found {len(img_files)} total images, {len(labeled_files)} already labeled")
    print(f"Remaining unlabeled: {len(unlabeled_files)}")
    
    if args.prioritize:
        # Load CNN predictions
        predictions = load_cnn_predictions(args.predictions)
        print(f"Loaded predictions for {len(predictions)} images")
        
        # Get prioritized images (with caching)
        prioritized_images = get_prioritized_images_cached(
            unlabeled_files, predictions, labeled_files, 
            args.predictions, force_refresh=args.force_refresh
        )
        
        if args.show_summary:
            show_priority_summary(prioritized_images, num_show=50)
            return
        
        # Show top priorities summary
        show_priority_summary(prioritized_images, num_show=min(20, len(prioritized_images)))
        
        # Select top priority images for annotation
        sample_size = min(args.samples, len(prioritized_images))
        sampled_files = [img_info[0] for img_info in prioritized_images[:sample_size]]
        
        print(f"\n🎯 Selected {sample_size} highest priority images for annotation")
        
    else:
        # Random sampling (original behavior)
        sample_size = min(args.samples, len(unlabeled_files))
        sampled_files = random.sample(unlabeled_files, sample_size)
        print(f"Selected {sample_size} random images for annotation")
    
    if not sampled_files:
        print("No images available for annotation!")
        return
    
    # Save status
    with open('level_label_status.json', 'w', encoding='utf-8') as f:
        json.dump({
            'labeled': sorted(list(labeled_files)),
            'unlabeled': sorted(unlabeled_files),
            'current_batch': sampled_files
        }, f, indent=2)
    
    # Load images for annotation
    print(f"\nLoading {len(sampled_files)} images for annotation...")
    images = []
    valid_files = []
    
    for filename in sampled_files:
        img_path = os.path.join(IMG_DIR, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            valid_files.append(filename)
        else:
            print(f"Warning: Could not load {filename}")
    
    if not images:
        print("No valid images loaded!")
        return
    
    print(f"Successfully loaded {len(images)} images")
    
    # Show first image preview
    if images:
        preview_img = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(preview_img)
        plt.title(f"Preview: {valid_files[0]}")
        plt.axis('off')
        plt.show()
    
    # Start annotation process
    print("\n" + "="*60)
    print("🚀 STARTING COFFEE LEVEL ANNOTATION")
    print("🎯 Focus: High-priority images for better class balance")
    print("📋 Instructions:")
    print("   - Use ← → arrows to select coffee level (0-10)")
    print("   - Press 'q' to save and continue")
    print("   - Press 'esc' to skip image")
    print("   - Close window to exit")
    print("="*60 + "\n")
    
    # Start annotation for each image
    for i, (img, filename) in enumerate(zip(images, valid_files)):
        print(f"Processing image {i+1}/{len(images)}: {filename}")
        
        # Show CNN prediction if available
        predictions = load_cnn_predictions(args.predictions)
        normalized_name = normalize_filename(filename)
        pred_info = None
        
        for pred_filename in predictions:
            if (pred_filename == filename or 
                pred_filename == normalized_name or
                normalize_filename(pred_filename) == normalized_name):
                pred_info = predictions[pred_filename]
                break
        
        if pred_info:
            print(f"  🤖 CNN Prediction: Level {pred_info['predicted_level']} "
                  f"(confidence: {pred_info['confidence']:.3f})")
        
        # Create selector for this image and get user selection
        try:
            selector = CoffeeLevelSelector(img, filename)
            level = selector.show()
        except Exception as e:
            print(f"  ❌ Error with selector: {e}")
            level = None
        
        if level is not None:
            # Save annotation
            output_file = os.path.join(manual_levels_dir, f"level_{filename}.json")
            os.makedirs(manual_levels_dir, exist_ok=True)
            
            annotation = {
                "filename": filename,
                "coffee_level": level,
                "timestamp": datetime.now().isoformat(),
                "annotator": "manual_smart",
                "version": "1.0"
            }
            
            if pred_info:
                annotation["cnn_prediction"] = pred_info['predicted_level']
                annotation["cnn_confidence"] = pred_info['confidence']
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, indent=2)
            
            print(f"  ✅ Saved: Level {level}")
        else:
            print(f"  ⏭️ Skipped")
    
    print(f"\n🎉 Annotation session complete!")
    print(f"📊 Check manual_levels/ directory for saved annotations")


if __name__ == "__main__":
    main()