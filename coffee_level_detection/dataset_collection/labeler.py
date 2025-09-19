# Semi-Automatic Coffee Level Annotation Workflow
# Select coffee cup counts (0-10) for training a classification model!
# 1. Imports
import cv2, json, os, random, sys
import matplotlib.pyplot as plt
from coffee_level_detection.dataset_collection.models import CoffeeLevelSelector
from tqdm import tqdm
# 2. Image Loader
IMG_DIR = 'processed_images'  # Your images directory

# Filter out images that are already labeled (present in manual_levels)
img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
manual_levels_dir = 'manual_levels'

labeled_files = set()
if os.path.exists(manual_levels_dir):
    for level_file in os.listdir(manual_levels_dir):
        if level_file.startswith('level_') and level_file.endswith('.json'):
            # Remove 'level_' prefix and '.json' suffix to get original filename
            original_name = level_file[6:-5]  # Remove 'level_' and '.json'
            labeled_files.add(original_name)
# Remove already labeled files from img_files
img_files = [f for f in img_files if f not in labeled_files]

# Save labeled and unlabeled filenames to JSON
all_img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
unlabeled_files = [f for f in all_img_files if f not in labeled_files]
with open('level_label_status.json', 'w', encoding='utf-8') as f:
    json.dump({
        'labeled': sorted(list(labeled_files)),
        'unlabeled': sorted(unlabeled_files)
    }, f, indent=2)

# Take a random sample of 50 files (or fewer if there are less than 50)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, default=10, help='Number of images to process')
args = parser.parse_args()

sample_size = min(args.samples, len(img_files))
sampled_files = random.sample(img_files, sample_size)

# Load the sampled images
images = [cv2.imread(os.path.join(IMG_DIR, f)) for f in sampled_files]
print(f"Loaded {len(images)} images.")

# Show the first image preview
if images:
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    plt.title(f"Starting annotation session with {len(images)} images")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)  # Show for 2 seconds
    plt.close()

# 3. Coffee Level Selection

# 4. Manual Coffee Level Annotation
# Select coffee cup counts (0-10) for each image.
coffee_levels = []
for filename, img in tqdm(zip(sampled_files, images), desc="Annotating coffee levels"):
    try:
        print(f"\nAnnotating image: {filename}")
        selector = CoffeeLevelSelector(img, filename)
        level = selector.show()
        
        if selector.exit_requested:
            print("Annotation session ended by user.")
            break
        
        if level is not None:
            coffee_levels.append({
                'filename': filename,
                'coffee_level': level
            })
            print(f"Saved: {filename} -> {level} cups")
        else:
            print(f"Skipped: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

# 5. Save Coffee Level Annotations
# These become your training labels for classification.
if coffee_levels:
    import datetime
    
    # Create output directory
    os.makedirs('manual_levels', exist_ok=True)
    
    # Save individual annotation files
    for annotation in coffee_levels:
        filename = annotation['filename']
        level = annotation['coffee_level']
        
        # Add timestamp
        timestamp = datetime.datetime.now().isoformat()
        annotation_data = {
            'filename': filename,
            'coffee_level': level,
            'timestamp': timestamp,
            'annotator': 'manual',
            'version': '1.0'
        }
        
        # Save as JSON file
        base_name = os.path.splitext(filename)[0]
        json_filename = f'manual_levels/level_{base_name}.json'
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2)
    
    # Save summary file
    summary_data = {
        'total_annotated': len(coffee_levels),
        'session_timestamp': datetime.datetime.now().isoformat(),
        'level_distribution': {},
        'annotations': coffee_levels
    }
    
    # Calculate level distribution
    for annotation in coffee_levels:
        level = annotation['coffee_level']
        summary_data['level_distribution'][str(level)] = summary_data['level_distribution'].get(str(level), 0) + 1
    
    with open('coffee_level_annotations_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n✅ Coffee level annotations saved!")
    print(f"📁 Individual files: manual_levels/ ({len(coffee_levels)} files)")
    print(f"📊 Summary file: coffee_level_annotations_summary.json")
    print(f"\nLevel distribution:")
    for level, count in sorted(summary_data['level_distribution'].items(), key=lambda x: int(x[0])):
        cups_text = "cups" if int(level) != 1 else "cup"
        print(f"  {level} {cups_text}: {count} images")
else:
    print("No annotations were saved.")

print("\n🎯 Next steps:")
print("- Use the annotated data to train a coffee level classification model")
print("- Consider using frameworks like PyTorch, scikit-learn, or TensorFlow")
print("- The JSON files contain structured data perfect for ML training pipelines")
