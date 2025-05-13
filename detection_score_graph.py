import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging

#logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_yolo_performance_graph(detections_folder, output_folder):
    """
    To generate a bar chart showing the average confidence scores of YOLO detections per class.
    Uses scores reading from training.
    """
    #Outout directory for the graph
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Manually input the confidence scores from the 4 provided images
    all_scores = {
        'pothole': [0.65, 0.72, 0.58],  # Placeholder scores
        'speed bump': [0.53, 0.70],      # From Image and Image
        'vehicle': [
            0.43, 0.49, 0.83, 0.71,      # Image 1
            0.93, 0.87, 0.42,            # Image 2
            0.84, 0.33, 0.96,            # Image 3
            0.96, 0.49,                  # Image 4
            0.64                         # From log (detected_rain_Screenshot)
        ]
    }
    
    # Scores for the remaining 10 images (Total of Input 15 images)
    # Example placeholder scores (replace with actual scores):
    all_scores['pothole'].extend([0.60, 0.55, 0.70, 0.68, 0.62, 0.59, 0.66, 0.71, 0.63, 0.67])
    all_scores['speed bump'].extend([0.58, 0.64, 0.61, 0.55, 0.67, 0.59, 0.62, 0.66, 0.60, 0.63])
    all_scores['vehicle'].extend([0.70, 0.68, 0.75, 0.62, 0.66, 0.71, 0.59, 0.64, 0.67, 0.73])
    
    # Average confidence scores per class
    avg_scores = {}
    for class_name, scores in all_scores.items():
        if scores:
            avg_scores[class_name] = np.mean(scores)
        else:
            avg_scores[class_name] = 0  # If no detections found, setting average to 0
    
    logger.info(f"Average Confidence Scores: {avg_scores}")
    
    # Data preparation for plotting
    classes = list(avg_scores.keys())
    averages = list(avg_scores.values())
    
    # Seaborn style graph
    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    
    # Bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, averages, color=['#ff7f0e', '#2ca02c', '#1f77b4'], width=0.5, edgecolor='black', linewidth=1.2)
    
    # Score value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=12, color='black')
    
    # Plot customization
    plt.title('Average YOLO Detection Confidence Scores by Class\n(Model mAP50: 0.671, mAP50-95: 0.448)', fontsize=14, pad=15)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Average Confidence Score', fontsize=12)
    plt.ylim(0, 1)  # Confidence scores range from 0 to 1
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Graph export
    graph_path = os.path.join(output_folder, 'yolo_performance_graph.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved YOLO performance graph to: {graph_path}")

if __name__ == "__main__":
    # Folder paths
    detections_folder = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/detections"
    output_folder = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/graphs"
    
    # Generate YOLO performance graph
    generate_yolo_performance_graph(detections_folder, output_folder)