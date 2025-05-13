import cv2
import os
import numpy as np
from pathlib import Path
import random  # For dummy detection

# Advanced enhancement functions
def dehaze_image(image):
    alpha = 1.8
    beta = 25
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def derain_image(image):
    kernel = np.ones((3, 3), np.uint8)
    derained = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cv2.bilateralFilter(derained, 9, 75, 75)

def desnow_image(image):
    snow_removed = cv2.medianBlur(image, 7)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(snow_removed, -1, kernel)
    return cv2.convertScaleAbs(sharpened, alpha=1.3, beta=15)

def enhance_night_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

# Dummy condition detection
def detect_conditions(image, model=None):  # Model param kept for compatibility
    # For testing, randomly picked conditions(Manually)
    all_conditions = ["rain", "snow", "fog", "night"]
    conditions = random.sample(all_conditions, k=random.randint(0, 2))  # 0-2 conditions
    return conditions if conditions else ["normal"]

# Enhance image based on detected conditions
def enhance_image(image, conditions):
    enhanced_image = image.copy()
    for condition in conditions:
        if condition == "fog":
            enhanced_image = dehaze_image(enhanced_image)
        elif condition == "rain":
            enhanced_image = derain_image(enhanced_image)
        elif condition == "snow":
            enhanced_image = desnow_image(enhanced_image)
        elif condition == "night":
            enhanced_image = enhance_night_image(enhanced_image)
    return enhanced_image

# Process images and saving to a single folder
def process_images(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    valid_extensions = ('.jpg', '.jpeg', '.png')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load {filename}")
                continue

            # Detect conditions (dummy for now)
            conditions = detect_conditions(image)
            print(f"{filename} - Detected conditions: {conditions}")

            # Enhance image
            enhanced_image = enhance_image(image, conditions)

            # Create output filename with condition tags
            base_name, ext = os.path.splitext(filename)
            condition_tag = "_".join(conditions) if conditions else "normal"
            output_filename = f"enhanced_{condition_tag}_{base_name}{ext}"
            output_path = os.path.join(output_folder, output_filename)

            # Save enhanced image
            cv2.imwrite(output_path, enhanced_image)
            print(f"Saved to: {output_path}")

# Main execution
if __name__ == "__main__":
    input_folder = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/input_images"
    output_folder = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/output_images"  # Single output folder
    process_images(input_folder, output_folder)