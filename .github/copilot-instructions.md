# Coffee Level Detection - AI Coding Guidelines

## Project Overview
This is a computer vision project for detecting coffee levels in images using manual annotation and machine learning. The project follows a modular architecture with a clear data pipeline from raw images to trained models for coffee cup counting (0-10 cups).

## Architecture & Data Flow

### Core Pipeline
1. **Raw Images** → `dataset_collection/labeler.py` → **Coffee Level Annotations**
2. **Coffee Level Annotations** → `dataset_collection/image_splitter.py` → **Processed Images** 
3. **Processed Images** → `training/` → **Trained Models**
4. **Trained Models** → `inference/` → **Predictions**

### Directory Structure
- `coffee_level_detection/` - Main package with modular components
  - `dataset_collection/` - Image annotation and dataset preparation
  - `preprosessing/` - Data preprocessing utilities (currently empty)  
  - `training/` - Model training pipeline (currently empty)
  - `inference/` - Model inference and prediction (currently empty)
- `manual_levels/` - Contains coffee level annotations as JSON files
- `masked_images/` - Contains masked coffee images (gitignored)
- `processed_images/` - Split and augmented images (gitignored)

## Key Development Patterns

### Coffee Level Annotation Workflow
The project uses an interactive coffee cup counting approach in `dataset_collection/labeler.py`:
- Samples random images from `raw_img/` directory
- Uses `CoffeeLevelSelector` for keyboard-driven level selection (0-10 cups)
- Navigation: left/right arrows cycle through levels, 'q' saves, 'esc' skips
- Tracks labeling progress in `level_label_status.json`
- Saves annotations to `manual_levels/` directory as JSON files
- Supports `--samples` argument to control batch size (default: 10)

### Interactive Selection Interface
`CoffeeLevelSelector` class provides:
- Matplotlib-based image display with keyboard controls
- Visual feedback showing current selection (0-10 cups)
- Looping navigation (0→1→...→10→0)
- Clean exit handling and progress tracking
- Detailed instructions overlay on image

### Image Processing Convention
`image_splitter.py` follows a specific augmentation pattern:
- Splits images vertically at midpoint (left/right halves)
- Creates 4 variants per input: `left_`, `right_`, `left_flipped_`, `right_flipped_`
- Uses vertical flipping (`cv2.flip(img, 0)`) for data augmentation
- Default input/output: `./masked_images` → `./processed_images`

### File Naming Conventions
- Raw images: Standard image formats in `raw_img/`
- Coffee level annotations: `level_{base_filename}.json` in `manual_levels/`
- Processed images: `{split}_{transformation}_{original_filename}`
- Summary data: `coffee_level_annotations_summary.json`

## Annotation Data Format

### Individual Annotation Files (`manual_levels/level_*.json`)
```json
{
  "filename": "photo_2018-01-15_07-38-04.jpg",
  "coffee_level": 3,
  "timestamp": "2025-09-19T10:30:45.123456",
  "annotator": "manual",
  "version": "1.0"
}
```

### Summary File (`coffee_level_annotations_summary.json`)
```json
{
  "total_annotated": 25,
  "session_timestamp": "2025-09-19T10:30:45.123456",
  "level_distribution": {"0": 5, "1": 8, "2": 7, "3": 5},
  "annotations": [...]
}
```

## Development Environment

### Dependencies & Tools
- **Poetry** for dependency management (pyproject.toml)
- **OpenCV** (`opencv-python`) for image processing
- **Matplotlib** for visualization and interactive annotation
- **scikit-learn** for ML utilities
- **tqdm** for progress bars
- Python 3.11-3.13 compatibility

### Key Commands
```bash
# Install dependencies
poetry install

# Run coffee level annotation tool
python -m coffee_level_detection.dataset_collection.labeler --samples 20

# Process annotated images
python -m coffee_level_detection.dataset_collection.image_splitter --input_dir ./masked_images --output_dir ./processed_images
```

### Module Execution Pattern
All modules support `python -m` execution:
- `python -m coffee_level_detection.dataset_collection.labeler`
- `python -m coffee_level_detection.dataset_collection.image_splitter`

## Current Development State
- ✅ **Dataset Collection**: Fully implemented with interactive coffee level selection
- ✅ **Image Processing**: Split and flip augmentation pipeline  
- 🚧 **Preprocessing**: Module structure exists, implementation pending
- 🚧 **Training**: Module structure exists, implementation pending
- 🚧 **Inference**: Module structure exists, implementation pending

## Working with Annotations
- Raw images should be placed in `raw_img/` directory
- The system automatically excludes already-annotated images
- Use `level_label_status.json` to track annotation progress
- Individual JSON files in `manual_levels/` contain structured annotation data
- Summary file provides session overview and level distribution statistics
- Image formats: `.jpg`, `.png` (case-insensitive)

## Code Style & Documentation Notes
- Import grouping: standard library, third-party, local modules
- Use f-strings for formatting
- Prefer `os.path.join()` for cross-platform paths
- Interactive tools use matplotlib with keyboard event handling
- Command-line interfaces use argparse with sensible defaults
- JSON files for structured data storage with timestamps and metadata
- **All functions, classes, and modules must include clear, concise docstrings** (Google or NumPy style recommended)
- **README.md must be professional, with project overview, setup, usage, and architecture details**
- **Modules should include top-level docstrings describing their purpose and usage**