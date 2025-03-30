import cv2
from ultralytics import YOLO
import os
from tqdm import tqdm
import logging

class ImageSegmenter:
    def __init__(self, model_path='yolov8n-seg.pt', confidence_threshold=0.5):
        """Initialize the image segmenter with YOLO model"""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def segment_images(self, input_dir, output_dir):
        """
        Segment a series of images using YOLO
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save segmented images
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get list of image files
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            image_files = [f for f in os.listdir(input_dir) 
                         if f.lower().endswith(image_extensions)]
            
            if not image_files:
                raise ValueError("No images found in the input directory")
            
            self.logger.info(f"Found {len(image_files)} images to process")
            
            # Process each image with progress bar
            for image_file in tqdm(image_files, desc="Segmenting images"):
                try:
                    # Read image
                    input_path = os.path.join(input_dir, image_file)
                    image = cv2.imread(input_path)
                    
                    if image is None:
                        self.logger.warning(f"Failed to read image: {image_file}")
                        continue
                    
                    # Perform segmentation
                    results = self.model(image, conf=self.confidence_threshold)
                    
                    # Process and save results
                    if results[0].masks is not None:
                        segmented_image = results[0].plot()
                        output_path = os.path.join(output_dir, f"seg_{image_file}")
                        cv2.imwrite(output_path, segmented_image)
                    else:
                        self.logger.warning(f"No objects segmented in {image_file}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {image_file}: {str(e)}")
                    continue
            
            self.logger.info("Image segmentation complete!")
            
        except Exception as e:
            self.logger.error(f"Error in segmentation process: {str(e)}")
            raise

def main():
    try:
        # Initialize segmenter
        segmenter = ImageSegmenter(
            model_path="yolov8n-seg.pt",
            confidence_threshold=0.5
        )
        
        # Process images
        segmenter.segment_images(
            input_dir="input_images",    #Path to input file
            output_dir="output_images"   #Path to output folder
        )
        
    except Exception as e:
        print(f"Failed to process images: {str(e)}")

if __name__ == "__main__":
    main()