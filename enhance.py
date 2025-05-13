import cv2
import os
import sys
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import warnings
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
import logging

warnings.filterwarnings("ignore", category=UserWarning)

# Path for Zero-DCE
sys.path.append("/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/Zero-DCE/Zero-DCE_code")
from model import enhance_net_nopool

# Logging for setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device (force CPU for now)
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# AOD-Net
class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=True)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=True)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2, bias=True)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=5, padding=2, bias=True)
        self.conv5 = nn.Conv2d(9, 3, kernel_size=5, padding=2, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.conv3(concat1))
        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.conv4(concat2))
        concat3 = torch.cat((x1, x2, x4), 1)
        k = self.relu(self.conv5(concat3))
        output = k * x - k + 1.0
        return torch.clamp(output, 0, 1)

class ConditionClassifier(nn.Module):
    def __init__(self):
        super(ConditionClassifier, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, 4)  # fog, high beam, night, rain
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

def load_classifier(model_path):
    model = ConditionClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_aod_net(model_path):
    model = AODNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_zero_dce(model_path):
    model = enhance_net_nopool()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

enhance_transform = transforms.Compose([
    transforms.ToTensor(),
])

def compute_ssim(original, enhanced):
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    score = ssim(gray_orig, gray_enh, data_range=255)
    return score

def dehaze_image(image, aod_model):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_tensor = enhance_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        dehazed_tensor = aod_model(img_tensor)
    dehazed = dehazed_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    dehazed = (dehazed * 255).astype(np.uint8)
    dehazed_bgr = cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)
    # Edge preservation and sharpening
    blurred = cv2.GaussianBlur(dehazed_bgr, (5, 5), 0)
    sharpened = cv2.addWeighted(dehazed_bgr, 1.5, blurred, -0.5, 0)
    return sharpened

def derain_image(image, use_learning=False):
    if use_learning:
        # Placeholder for learning-based deraining (e.g., using a pretrained model)
        # DDN or RainNet could be integrate 
        return image  # Replacing with actual implementation
    else:
        # Improved non-learning deraining
        blurred = cv2.GaussianBlur(image, (7, 7), 0)
        enhanced = cv2.addWeighted(image, 1.8, blurred, -0.8, 0)
        return cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)  # Boost contrast

def enhance_night_image(image, zero_dce_model):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_tensor = enhance_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        _, enhanced_tensor, _ = zero_dce_model(img_tensor)
    enhanced = enhanced_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    enhanced = (enhanced * 255).astype(np.uint8)
    return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

def postprocess_adverse(image, conditions):
    """Additional preprocessing for adverse conditions to aid detection."""
    if "fog" in conditions or "night" in conditions or "rain" in conditions:
        # Adaptive histogram equalization for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return clahe.apply(image)
    return image

def detect_conditions(image, model):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(img_tensor).squeeze().cpu().numpy()
    conditions = []
    labels = ["fog", "high beam", "night", "rain"]
    threshold = 0.5
    for i, prob in enumerate(preds):
        if prob > threshold:
            conditions.append(labels[i])
    return conditions if conditions else ["normal"]

def enhance_image(image, conditions, aod_model, zero_dce_model):
    enhanced_image = image.copy()
    if "fog" in conditions:
        enhanced_image = dehaze_image(enhanced_image, aod_model)
    elif "night" in conditions and "rain" not in conditions:
        enhanced_image = enhance_night_image(enhanced_image, zero_dce_model)
    elif "rain" in conditions and "night" not in conditions:
        enhanced_image = derain_image(enhanced_image)
    elif "night" in conditions and "rain" in conditions:
        derained = derain_image(enhanced_image)
        enhanced_image = enhance_night_image(derained, zero_dce_model)
    # Apply postprocessing for adverse conditions
    enhanced_image = postprocess_adverse(enhanced_image, conditions)
    return enhanced_image

def calculate_distance(bbox, image_width, object_type="vehicle", focal_length=4.0, sensor_width=5.6):
    # Object real-world width (meters)
    object_width_m = {"vehicle": 1.8, "pothole": 0.5, "speed bump": 1.0}.get(object_type, 1.0)
    bbox_width_px = bbox[2] - bbox[0]
    if bbox_width_px == 0:  # Avoid division by zero
        return float('inf')
    distance_m = (focal_length * object_width_m * image_width) / (bbox_width_px * sensor_width)
    return distance_m

def process_images(input_folder, processed_folder, detections_folder, classifier, aod_model, zero_dce_model, yolo_model):
    Path(processed_folder).mkdir(parents=True, exist_ok=True)
    Path(detections_folder).mkdir(parents=True, exist_ok=True)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)
            if image is None:
                logger.error(f"Failed to load {filename}")
                continue

            # Step 1: Enhance the image
            conditions = detect_conditions(image, classifier)
            enhanced_image = enhance_image(image, conditions, aod_model, zero_dce_model)
            ssim_score = compute_ssim(image, enhanced_image)

            logger.info(f"{filename} - Detected conditions: {conditions}, SSIM: {ssim_score:.4f}")

            # Save enhanced image
            base_name, ext = os.path.splitext(filename)
            condition_tag = "_".join(conditions) if conditions else "normal"
            processed_filename = f"enhanced_{condition_tag}_{base_name}{ext}"
            processed_path = os.path.join(processed_folder, processed_filename)
            cv2.imwrite(processed_path, enhanced_image)
            logger.info(f"Saved enhanced image to: {processed_path}")

            # Step 2: YOLOv8 object detection with lower confidence for adverse conditions
            conf_threshold = 0.3 if any(c in conditions for c in ["fog", "night", "rain"]) else 0.5
            results = yolo_model.predict(processed_path, conf=conf_threshold, iou=0.45)
            detection_image = enhanced_image.copy()

            # Step 3: Calculate distances and draw bounding boxes
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for box, label, conf in zip(boxes, labels, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    label_name = result.names[int(label)]
                    conf_score = conf

                    # Calculate distance
                    distance = calculate_distance([x1, y1, x2, y2], image.shape[1], object_type=label_name.lower())
                    
                    # Draw bounding box and label
                    cv2.rectangle(detection_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{label_name}: {conf_score:.2f}, {distance:.2f}m"
                    cv2.putText(detection_image, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save detection image
            detection_filename = f"detected_{condition_tag}_{base_name}{ext}"
            detection_path = os.path.join(detections_folder, detection_filename)
            cv2.imwrite(detection_path, detection_image)
            logger.info(f"Saved detection image to: {detection_path}")

if __name__ == "__main__":
    input_folder = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/input_images"
    processed_folder = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/processed_images"
    detections_folder = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/detections"
    classifier_path = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/models/condition_classifier.pth"
    aod_model_path = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/pretrained/aod_net_trained.pth"
    zero_dce_model_path = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/Zero-DCE/Zero-DCE_code/snapshots/model.pth"
    yolo_model_path = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/models/Object_detection_Yolo.pt"
    
    classifier = load_classifier(classifier_path)
    aod_model = load_aod_net(aod_model_path)
    zero_dce_model = load_zero_dce(zero_dce_model_path)
    yolo_model = YOLO(yolo_model_path)
    
    process_images(input_folder, processed_folder, detections_folder, classifier, aod_model, zero_dce_model, yolo_model)