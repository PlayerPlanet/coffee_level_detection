# Coffee Level Detection

Coffee Level Detection is a computer vision project for detecting and counting coffee levels in images using manual annotation and deep learning. The project is modular, with a clear data pipeline from raw images to trained models and inference.

## Features
- Interactive annotation tool for coffee level labeling
- Automated image splitting and augmentation
- Deep learning model training (coffeeCNN)
- Inference and prediction modules
- Structured annotation and summary files

## Directory Structure
```
coffee_level_detection/
    dataset_collection/   # Annotation and dataset preparation
    preprosessing/        # Data preprocessing utilities
    training/             # Model training pipeline
    inference/            # Model inference and prediction
manual_levels/            # Coffee level annotation JSON files
masked_images/            # Masked coffee images
processed_images/         # Augmented images
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PlayerPlanet/coffee_level_detection.git
   ```
2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

## Usage

### Annotate Coffee Levels
```bash
python -m coffee_level_detection.dataset_collection.labeler --samples 20
```

### Process Images
```bash
python -m coffee_level_detection.dataset_collection.image_splitter --input_dir ./masked_images --output_dir ./processed_images
```

### Train Model
```bash
python -m coffee_level_detection.training.train_coffeeCNN --f compiled_coffee_level_annotations.json --batch 10 --epochs 100 --img_dir processed_images
```

### Inference
```bash
python -m coffee_level_detection.inference.predict --model_path <model.pt> --img_dir <images>
```

## Data Format
- Annotation files: `manual_levels/level_*.json`
- Summary: `coffee_level_annotations_summary.json`
- Processed images: `processed_images/`

## Development Standards
- All functions and classes must have docstrings
- Professional documentation and code comments
- Modular, maintainable codebase

## License
MIT License
