"""
Script to visualize N images with a specific coffee level label from inference results.
"""
import os
import json
import argparse
import matplotlib.pyplot as plt
import cv2



def load_annotations(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('annotation_data', [])

def show_images_with_label(label, image_dir, json_path, n=5):
    annotations = load_annotations(json_path)
    filtered = [a for a in annotations if a['coffee_level'] == label]
    if not filtered:
        print(f"No images found with label {label}.")
        return
    for ann in filtered[:n]:
        img_path = os.path.join(image_dir, ann['filename'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f"{ann['filename']}\nCoffee Level: {label}")
        plt.axis('off')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize N images with a specific coffee level label.")
    parser.add_argument('--label', type=int, required=True, help='Coffee level label (0-10)')
    parser.add_argument('--n', type=int, default=5, help='Number of images to show')
    parser.add_argument('--image_dir', type=str, default='processed_images', help='Directory containing images')
    parser.add_argument('--annotations', type=str, default='inference_coffee_level_annotations.json', help='Path to inference annotations JSON')
    args = parser.parse_args()
    IMAGE_DIR = args.image_dir
    ANNOTATIONS_PATH = args.annotations
    show_images_with_label(args.label, IMAGE_DIR, ANNOTATIONS_PATH, args.n)

if __name__ == '__main__':
    main()
