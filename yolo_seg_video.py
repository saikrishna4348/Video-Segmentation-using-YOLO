import cv2
from ultralytics import YOLO
import os
from tqdm import tqdm
import logging

class VideoSegmenter:
    def __init__(self, model_path='yolov8n-seg.pt', confidence_threshold=0.5):
        """Initialize the video segmenter with YOLO model"""
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
    
    def process_video(self, video_path, output_dir, frame_skip=1):
        """
        Process video and save segmented frames
        
        Args:
            video_path (str): Input video path
            output_dir (str): Output directory for segmented frames
            frame_skip (int): Number of frames to skip between processing
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"Video info: {frame_count} frames, {fps} FPS, {width}x{height}")
            
            frame_num = 0
            processed_count = 0
            
            # Progress bar
            with tqdm(total=frame_count, desc="Processing frames") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every nth frame
                    if frame_num % frame_skip == 0:
                        # Perform segmentation
                        results = self.model(frame, conf=self.confidence_threshold)
                        
                        if results[0].masks is not None:
                            segmented_frame = results[0].plot()
                            output_path = os.path.join(output_dir, f"frame_{processed_count:06d}.jpg")
                            cv2.imwrite(output_path, segmented_frame)
                            processed_count += 1
                    
                    frame_num += 1
                    pbar.update(1)
            
            cap.release()
            self.logger.info(f"Processing complete! Saved {processed_count} segmented frames")
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            raise

def main():
    try:
        # Initialize segmenter
        segmenter = VideoSegmenter(
            model_path="yolov8n-seg.pt",
            confidence_threshold=0.5
        )
        
        # Process video
        segmenter.process_video(
            video_path="sample_video.mp4",
            output_dir="video_output",
            frame_skip=1  
        )
        
    except Exception as e:
        print(f"Failed to process video: {str(e)}")

if __name__ == "__main__":
    main()