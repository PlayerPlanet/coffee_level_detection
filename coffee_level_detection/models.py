from transformers import pipeline
import numpy as np
# Use the pretrained checkpoint name
checkpoint = "facebook/mask2former-swin-small-coco-instance"

# The pipeline will automatically load the correct processor and model
segmenter = pipeline("image-segmentation", model=checkpoint, use_fast=True)

def coffee_fill_percentage(image_path):
    segments = segmenter(image_path)
    mask_pot = None
    mask_coffee = None
    for seg in segments:
        if "pot" in seg["label"]:
            mask_pot = np.array(seg["mask"])
        if "coffee" in seg["label"]:
            mask_coffee = np.array(seg["mask"])
    if mask_pot is None or mask_coffee is None:
        return None
    return 100 * np.sum(mask_coffee) / np.sum(mask_pot)

