import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from typing import List, Dict

class YOLOSegmentationMetrics:
    def __init__(self, model_path: str = 'yolov8n-seg.pt', image_dir: str = '.', 
                 gt_dir: str = '.', conf_threshold: float = 0.5):
        """
        Initialize evaluator with YOLO model and directories
        
        Args:
            model_path: Path to YOLO model
            image_dir: Directory with original images
            gt_dir: Directory with ground truth masks
            conf_threshold: Confidence threshold for predictions
        """
        self.model = YOLO(model_path)
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.conf_threshold = conf_threshold

    def load_and_predict(self) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Load images, predict masks, and load ground truth"""
        image_files = sorted([f for f in os.listdir(self.image_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))])
        gt_files = sorted([f for f in os.listdir(self.gt_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(image_files) != len(gt_files):
            raise ValueError("Number of images and ground truth masks must match")
        
        pred_masks = []
        gt_masks = []
        
        for img_file, gt_file in zip(image_files, gt_files):
            # Load and predict
            img_path = os.path.join(self.image_dir, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Could not load {img_file}")
                continue
                
            results = self.model(image, conf=self.conf_threshold)
            
            # Process prediction
            if results[0].masks is not None:
                mask = results[0].masks.data[0].cpu().numpy()
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)  # Binarize
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            # Load ground truth
            gt_path = os.path.join(self.gt_dir, gt_file)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            if gt is None:
                print(f"Warning: Could not load {gt_file}")
                continue
            gt = (gt > 0).astype(np.uint8)
            
            pred_masks.append(mask)
            gt_masks.append(gt)
        
        return pred_masks, gt_masks

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         smooth: float = 1e-6) -> Dict[str, float]:
        """Calculate all metrics for a single image pair"""
        # True positives, false positives, false negatives
        tp = np.sum(y_true * y_pred)
        fp = np.sum(y_pred * (1 - y_true))
        fn = np.sum(y_true * (1 - y_pred))
        tn = np.sum((1 - y_true) * (1 - y_pred))
        
        # Metrics
        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        iou = tp / (tp + fp + fn + smooth)
        f1 = (2 * precision * recall) / (precision + recall + smooth)
        
        return {
            'dice': dice,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'f1': f1
        }

    def evaluate(self) -> Dict[str, List[float]]:
        """Calculate metrics for all images"""
        pred_masks, gt_masks = self.load_and_predict()
        
        metrics = {
            'dice': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'iou': [],
            'f1': []
        }
        
        for pred, gt in zip(pred_masks, gt_masks):
            if pred.shape != gt.shape:
                raise ValueError("Prediction and ground truth shapes must match")
            
            result = self.calculate_metrics(gt, pred)
            for key in metrics:
                metrics[key].append(result[key])
        
        return metrics

    def plot_metrics(self, metrics: Dict[str, List[float]], save_path: str = None):
        """Plot all metrics"""
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        metric_names = ['dice', 'accuracy', 'precision', 'recall', 'iou', 'f1']
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        
        x = range(1, len(metrics['dice']) + 1)
        
        for ax, metric, color in zip(axes, metric_names, colors):
            ax.plot(x, metrics[metric], f'{color}.-', label=metric.capitalize())
            avg = np.mean(metrics[metric])
            ax.axhline(y=avg, color=color, linestyle='--', 
                      label=f'Avg: {avg:.3f}')
            ax.set_title(f'{metric.capitalize()} per Image')
            ax.set_xlabel('Image Number')
            ax.set_ylabel(metric.capitalize())
            ax.grid(True)
            ax.set_ylim(0, 1)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        plt.close()

def main():
    # Usage example
    evaluator = YOLOSegmentationMetrics(
        model_path="yolov8n-seg.pt",
        image_dir="input_images",  #Path to Original images
        gt_dir="annotations",  #Path to  Gold standard masks
        conf_threshold=0.5
    )
    
    # Calculate metrics
    metrics = evaluator.evaluate()
    
    # Print average metrics
    print("\nAverage Metrics:")
    for metric, values in metrics.items():
        print(f"{metric.capitalize()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Plot metrics
    evaluator.plot_metrics(metrics, save_path="segmentation_metrics.png")

if __name__ == "__main__":
    # Requirements: pip install numpy matplotlib opencv-python ultralytics
    main()