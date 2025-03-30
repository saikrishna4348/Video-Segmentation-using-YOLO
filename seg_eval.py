import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from typing import List, Dict

class YOLOConfidenceEvaluator:
    def __init__(self, model_path: str = 'yolov8n-seg.pt', image_dir: str = '.', 
                 conf_threshold: float = 0.5):
        """
        Initialize evaluator with YOLO model and image directory
        
        Args:
            model_path: Path to YOLO model
            image_dir: Directory containing images to process
            conf_threshold: Confidence threshold for predictions
        """
        self.model = YOLO(model_path)
        self.image_dir = image_dir
        self.conf_threshold = conf_threshold

    def get_confidence_scores(self) -> Dict[str, List[float]]:
        """Process images and extract confidence scores"""
        image_files = sorted([f for f in os.listdir(self.image_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if not image_files:
            raise ValueError("No images found in the directory")
        
        confidence_scores = []
        
        for img_file in image_files:
            # Load image
            img_path = os.path.join(self.image_dir, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Could not load {img_file}")
                continue
            
            # Get YOLO predictions
            results = self.model(image, conf=self.conf_threshold)
            
            # Extract confidence scores
            if results[0].boxes is not None and len(results[0].boxes.conf) > 0:
               
                conf = float(results[0].boxes.conf.max())
            else:
                conf = 0.0  
            
            confidence_scores.append(conf)
        
        return {
            'confidence': confidence_scores,
            'image_files': image_files  # For labeling
        }

    def plot_confidence(self, metrics: Dict[str, List[float]], save_path: str = None):
        """Plot confidence scores with min and max in legend"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(1, len(metrics['confidence']) + 1)
        conf_scores = metrics['confidence']
        
        # Calculate statistics
        avg_conf = np.mean(conf_scores)
        min_conf = np.min(conf_scores)
        max_conf = np.max(conf_scores)
        
        # Plot confidence scores
        ax.plot(x, conf_scores, 'm.-', label='Confidence')
        ax.set_title('YOLO Confidence Scores per Image')
        ax.set_xlabel('Image Number')
        ax.set_ylabel('Confidence Score')
        ax.grid(True)
        ax.set_ylim(0, 1)
        
        # Add average line
        ax.axhline(y=avg_conf, color='m', linestyle='--', 
                  label=f'Avg: {avg_conf:.3f}\nMin: {min_conf:.3f}\nMax: {max_conf:.3f}')
        
        
        ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8)
        
       
        if len(metrics['image_files']) <= 20:  # Avoid clutter with many images
            plt.xticks(x, metrics['image_files'], rotation=45, ha='right')
        else:
            plt.xticks(x[::max(1, len(x)//10)])  # Show every nth label for many images
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        plt.close()

    def evaluate_and_plot(self, save_path: str = "confidence_plot.png"):
        """Run evaluation and plot results"""
        metrics = self.get_confidence_scores()
        
        
        
        self.plot_confidence(metrics, save_path)

def main():
    
    evaluator = YOLOConfidenceEvaluator(
        model_path="yolov8n-seg.pt",
        image_dir="input_images",  
        conf_threshold=0.5
    )
    
    evaluator.evaluate_and_plot(save_path="yolo_confidence_plot.png")

if __name__ == "__main__":
    
    main()