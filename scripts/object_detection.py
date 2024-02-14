from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
from pathlib import Path
import cv2
import os
from logger import logger

class ObjectDetection:
    def __init__(self) -> None:
        """
        Initialize ObjectDetection class with YOLOv3 model pretrained on VOC dataset.
        """
        try:
            self.net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
            logger.info("Object detection model created")
        except Exception as e:
            logger.error(f"Error while creating object detection: {e}")
        
        
    def detect_from_image(self, image_path: str) -> tuple:
        """
        Perform object detection on a single image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple: Tuple containing detected class IDs, scores, bounding boxes, and the image.
        """
        try:
            x, img = data.transforms.presets.yolo.load_test(image_path, short=512)
            class_IDs, scores, bounding_boxs = self.net(x)
            logger.info("Object detected from image successfully")
            return class_IDs, scores, bounding_boxs, img
        except Exception as e:
            logger.error(f"Error while detecting objects from image: {e}")
            return (), (), (), None
    
    def detect_from_video(self, video_path: str) -> dict:
        """
        Perform object detection on a video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            dict: Dictionary containing detected class IDs, scores, bounding boxes, and images for each frame.
        """
        try:
            result = {'class_IDs':[], 'scores':[], 'bounding_boxs':[], 'img':[]}
            
            # Opens the Video file
            cap = cv2.VideoCapture(video_path)
            i = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                ps = (frame>0).sum()
                tp = frame.shape[0]*frame.shape[1]
                cv2.imwrite('frame'+str(i)+'.jpg', frame)
                x, img = data.transforms.presets.yolo.load_test('frame'+str(i)+'.jpg', short=512)
                class_IDs, scores, bounding_boxs = self.net(x)
                
                result['class_IDs'].append(class_IDs)
                result['scores'].append(scores)
                result['bounding_boxs'].append(bounding_boxs) 
                result['img'].append(img)
                if os.path.isfile('frame'+str(i)+'.jpg'):
                    os.remove('frame'+str(i)+'.jpg')
                i += 1
            cap.release()
            cv2.destroyAllWindows()
            
            logger.info("Object detection completed on video successfully")
            return result
        
        except Exception as e:
            logger.error(f"Error while detecting objects from video: {e}")
            return {}
        
    def plot_detection(self, img, class_IDs, scores, bounding_boxs) -> None:
        """
        Plot the object detection results on the image.

        Args:
            img: Image on which detection results will be plotted.
            class_IDs: IDs of detected classes.
            scores: Confidence scores of detected objects.
            bounding_boxs: Bounding boxes of detected objects.
        """
        try:
            ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                                class_IDs[0], class_names=self.net.classes)
            plt.show() 
            logger.info("Object detection plot generated successfully")
        
        except Exception as e:
            logger.error(f"Error while plotting object detection: {e}")

