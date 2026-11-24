import cv2
import mediapipe as mp
import time
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class VisionModels:
    def __init__(self, model_dir="models"):
        # Model paths
        self.classifier_path = os.path.join(
            model_dir, "efficientnet_lite0.tflite"
        )

        self.detector_path = os.path.join(
            model_dir, "efficientdet_lite0.tflite"
        )

        self.hand_path = os.path.join(
            model_dir, "hand_landmarker.task"
        )

        # Check model files
        for path in [
            self.classifier_path,
            self.detector_path,
            self.hand_path
        ]:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}"
                )

        # Load image classifier
        classifier_options = vision.ImageClassifierOptions(
            base_options=python.BaseOptions(
                model_asset_path=self.classifier_path
            ),
            max_results=3
        )

        self.classifier = vision.ImageClassifier.create_from_options(
            classifier_options
        )

        # Load object detector
        detector_options = vision.ObjectDetectorOptions(
            base_options=python.BaseOptions(
                model_asset_path=self.detector_path
            ),
            score_threshold=0.5,
            max_results=10
        )

        self.detector = vision.ObjectDetector.create_from_options(
            detector_options
        )

        # Load hand landmarker
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=self.hand_path
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(
            hand_options
        )

        print("All MediaPipe models loaded successfully.")

    def _create_image(self, rgb_frame):
        return mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

    def classify(self, rgb_frame):
        image = self._create_image(rgb_frame)

        start = time.perf_counter()
        result = self.classifier.classify(image)
        elapsed = (time.perf_counter() - start) * 1000

        return result, elapsed

    def detect_objects(self, rgb_frame):
        image = self._create_image(rgb_frame)

        start = time.perf_counter()
        result = self.detector.detect(image)
        elapsed = (time.perf_counter() - start) * 1000

        return result, elapsed

    def track_hands(self, rgb_frame):
        image = self._create_image(rgb_frame)

        start = time.perf_counter()
        result = self.hand_landmarker.detect(image)
        elapsed = (time.perf_counter() - start) * 1000

        return result, elapsed

    def close(self):
        self.classifier.close()
        self.detector.close()
        self.hand_landmarker.close()

        print("All MediaPipe models released.")
