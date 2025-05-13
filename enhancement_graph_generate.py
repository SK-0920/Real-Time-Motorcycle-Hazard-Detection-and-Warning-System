import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns  # For better aesthetics
from skimage.metrics import structural_similarity as ssim
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_mse(image1, image2):
    """
    Calculate Mean Squared Error (MSE) between two images.
    """
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def calculate_psnr(image1, image2, max_pixel_value=255.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))

def calculate_ssim(image1, image2):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    """
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def extract_base_name(filename, prefixes=None):
    """
    Extract the base name of a file by removing known prefixes and the extension.
    """
    if prefixes is None:
        prefixes = ['enhanced_high_beam_', 'enhanced_night_', 'enhanced_fog_', 'enhanced_']
    
    base_name = os.path.splitext(filename)[0]
    for prefix in prefixes:
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix):]
            break
    return base_name

def generate_metric_graphs(input_folder, enhanced_folder, output_folder):
    """
    Generate professional-looking graph images for average MSE, PSNR, and SSIM scores.
    """
    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    # Lists to store metrics for all image pairs
    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    
    # Get list of enhanced images
    enhanced_files = [f for f in os.listdir(enhanced_folder) if f.lower().endswith(valid_extensions)]
    enhanced_base_names = {extract_base_name(f): f for f in enhanced_files}
    
    # Iterate through input images
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_folder, filename)
            input_base_name = extract_base_name(filename)
            
            # Find the corresponding enhanced image
            enhanced_filename = None
            for base_name, enhanced_file in enhanced_base_names.items():
                if base_name == input_base_name:
                    enhanced_filename = enhanced_file
                    break
            
            if not enhanced_filename:
                logger.warning(f"Enhanced image not found for {filename}")
                continue
            
            enhanced_path = os.path.join(enhanced_folder, enhanced_filename)
            input_image = cv2.imread(input_path)
            enhanced_image = cv2.imread(enhanced_path)
            
            if input_image is None or enhanced_image is None:
                logger.error(f"Failed to load images: {input_path} or {enhanced_path}")
                continue
            
            if input_image.shape != enhanced_image.shape:
                logger.warning(f"Image dimensions do not match for {filename}, resizing enhanced image")
                enhanced_image = cv2.resize(enhanced_image, (input_image.shape[1], input_image.shape[0]))
            
            # Calculate metrics
            mse = calculate_mse(input_image, enhanced_image)
            psnr = calculate_psnr(input_image, enhanced_image)
            ssim_score = calculate_ssim(input_image, enhanced_image)
            
            mse_scores.append(mse)
            psnr_scores.append(psnr)
            ssim_scores.append(ssim_score)
            
            logger.info(f"Metrics for {filename}: MSE={mse:.2f}, PSNR={psnr:.2f}, SSIM={ssim_score:.4f}")
    
    if not mse_scores or not psnr_scores or not ssim_scores:
        logger.error("No valid image pairs found to calculate metrics.")
        return
    
    # Calculate average scores
    avg_mse = np.mean(mse_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    
    logger.info(f"Average Metrics: MSE={avg_mse:.2f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
    
    # Set seaborn style for professional look
    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    
    # Generate column chart for MSE
    plt.figure(figsize=(5, 6))
    bar = plt.bar(['Average MSE'], [avg_mse], color='#1f77b4', width=0.4, edgecolor='black', linewidth=1.2)
    plt.title('Average Mean Squared Error (MSE)', fontsize=14, pad=15)
    plt.ylabel('MSE', fontsize=12)
    plt.ylim(0, max(avg_mse * 1.2, 100))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    # Add value label on top of the bar
    plt.text(0, avg_mse + avg_mse * 0.05, f'{avg_mse:.2f}', ha='center', va='bottom', fontsize=12, color='black')
    mse_graph_path = os.path.join(output_folder, 'mse_graph.png')
    plt.savefig(mse_graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved MSE graph to: {mse_graph_path}")
    
    # Generate column chart for PSNR
    plt.figure(figsize=(5, 6))
    bar = plt.bar(['Average PSNR'], [avg_psnr], color='#2ca02c', width=0.4, edgecolor='black', linewidth=1.2)
    plt.title('Average Peak Signal-to-Noise Ratio (PSNR)', fontsize=14, pad=15)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.ylim(0, max(avg_psnr * 1.2, 40))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    # Add value label on top of the bar
    plt.text(0, avg_psnr + avg_psnr * 0.05, f'{avg_psnr:.2f}', ha='center', va='bottom', fontsize=12, color='black')
    psnr_graph_path = os.path.join(output_folder, 'psnr_graph.png')
    plt.savefig(psnr_graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved PSNR graph to: {psnr_graph_path}")
    
    # Generate column chart for SSIM
    plt.figure(figsize=(5, 6))
    bar = plt.bar(['Average SSIM'], [avg_ssim], color='#ff7f0e', width=0.4, edgecolor='black', linewidth=1.2)
    plt.title('Average Structural Similarity Index (SSIM)', fontsize=14, pad=15)
    plt.ylabel('SSIM', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    # Add value label on top of the bar
    plt.text(0, avg_ssim + 0.02, f'{avg_ssim:.4f}', ha='center', va='bottom', fontsize=12, color='black')
    ssim_graph_path = os.path.join(output_folder, 'ssim_graph.png')
    plt.savefig(ssim_graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SSIM graph to: {ssim_graph_path}")

if __name__ == "__main__":
    # Define folder paths
    input_folder = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/input_images"
    enhanced_folder = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/processed_images"
    output_folder = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/graphs"
    
    # Generate metric graphs
    generate_metric_graphs(input_folder, enhanced_folder, output_folder)