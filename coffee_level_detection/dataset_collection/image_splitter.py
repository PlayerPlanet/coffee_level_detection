import os
import cv2
import argparse
from tqdm import tqdm
# Create output directories if they don't exist

# Function to process images (split and flip vertically)
def process_images(input_folder, output_folder, image_extension):
    images = [f for f in os.listdir(input_folder) if f.endswith(image_extension)]
    transformed_images = []

    for image_name in tqdm(images, total=len(images)):
        img_path = os.path.join(input_folder, image_name)
        img = cv2.imread(img_path)

        # Ensure image is read and is in color
        if img is not None and len(img.shape) == 3:
            height, width = img.shape[:2]
            mid_width = width // 2

            # Splitting image vertically
            img_left = img[:, :mid_width]
            img_right = img[:, mid_width:]

            # Save split images
            cv2.imwrite(os.path.join(output_folder, f'left_{image_name}'), img_left)
            cv2.imwrite(os.path.join(output_folder, f'right_{image_name}'), img_right)

            # Flip right image vertically
            img_right_flipped = cv2.flip(img_right, 1)
            cv2.imwrite(os.path.join(output_folder, f'right_flipped_{image_name}'), img_right_flipped)
            img_left_flipped = cv2.flip(img_left, 1)  
            cv2.imwrite(os.path.join(output_folder, f'left_flipped_{image_name}'), img_left_flipped)


    return transformed_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./masked_images")
    parser.add_argument("--output_dir", type=str, default="./processed_images")
    args = parser.parse_args()
    input_folder = args.input_dir
    output_folder = args.output_dir
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    process_images(input_folder, output_folder, ".jpg")

