import cv2
import os

def add_synthetic_detections(image_path, output_path, detections, image_width):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    for obj in detections:
        x1, y1, x2, y2 = obj['bbox']
        label = obj['label']
        conf = obj['conf']
        # Calculate distance
        distance = calculate_distance([x1, y1, x2, y2], image_width, object_type=label.lower())
        # Draw bounding box (green, thickness 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add label text (green, above box)
        label_text = f"{label}: {conf:.2f}, {distance:.2f}m"
        cv2.putText(image, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(output_path, image)
    print(f"Saved synthetic detection image to: {output_path}")

# Reusing your calculate_distance function
def calculate_distance(bbox, image_width, object_type="vehicle", focal_length=4.0, sensor_width=5.6):
    object_width_m = {"vehicle": 1.8, "pothole": 0.5, "speed bump": 1.0}.get(object_type, 1.0)
    bbox_width_px = bbox[2] - bbox[0]
    if bbox_width_px == 0:
        return float('inf')
    distance_m = (focal_length * object_width_m * image_width) / (bbox_width_px * sensor_width)
    return distance_m

# Process each fog image with corrected bounding boxes
# Fog Image 1: Highway with multiple vehicles
fog_image1_path = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/processed_images/enhanced_fog_image1.png"
fog_image1_output = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/synthetic_detections/detected_fog_image1.png"
# Load image to get width
img1 = cv2.imread(fog_image1_path)
fog_image1_width = img1.shape[1] if img1 is not None else 640  # Default to 640 if loading fails
detections_fog1 = [
    {'bbox': (30, 50, 150, 200), 'label': 'vehicle', 'conf': 0.92},  # Truck on the left
    {'bbox': (200, 150, 350, 300), 'label': 'vehicle', 'conf': 0.95},  # Center car
    {'bbox': (450, 150, 600, 300), 'label': 'vehicle', 'conf': 0.90},  # Right car
]
add_synthetic_detections(fog_image1_path, fog_image1_output, detections_fog1, fog_image1_width)

# Fog Image 2: Road with motorcycle and truck
fog_image2_path = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/processed_images/enhanced_fog_image2.png"
fog_image2_output = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/synthetic_detections/detected_fog_image2.png"
img2 = cv2.imread(fog_image2_path)
fog_image2_width = img2.shape[1] if img2 is not None else 480  # Default to 480 if loading fails
detections_fog2 = [
    {'bbox': (200, 250, 280, 350), 'label': 'vehicle', 'conf': 0.82},  # Motorcycle
    {'bbox': (350, 150, 400, 200), 'label': 'vehicle', 'conf': 0.85},  # Truck in background
]
add_synthetic_detections(fog_image2_path, fog_image2_output, detections_fog2, fog_image2_width)

# Fog Image 3: Road with car
fog_image3_path = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/processed_images/enhanced_fog_image3.png"
fog_image3_output = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/synthetic_detections/detected_fog_image3.png"
img3 = cv2.imread(fog_image3_path)
fog_image3_width = img3.shape[1] if img3 is not None else 480  # Default to 480 if loading fails
detections_fog3 = [
    {'bbox': (150, 200, 300, 400), 'label': 'vehicle', 'conf': 0.96},  # Center car
]
add_synthetic_detections(fog_image3_path, fog_image3_output, detections_fog3, fog_image3_width)