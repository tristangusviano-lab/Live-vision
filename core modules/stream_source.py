import cv2


class StreamSource:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Could not open webcam.")

    def read(self):
        success, frame = self.camera.read()

        if not success or frame is None:
            return None

        return frame

    def release(self):
        if self.camera is not None:
            self.camera.release()