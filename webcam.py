import cv2
import numpy as np
import torch
import threading

class Webcam:
    def __init__(self, width=640, height=480, fps=30, shm=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.shm = shm
        self.is_streaming = False
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if self.shm is not None:
            self.image = np.ndarray((3, height, width), dtype=np.uint8, buffer=self.shm.buf)
        else:
            self.image = np.ndarray((3, height, width), dtype=np.uint8)
        threading.Thread(target=self.streaming).start()

    def process_image(self, image):
        image = image.transpose((2, 0, 1))  # Convert to CHW
        image = torch.from_numpy(image).float() / 255.0
        return image.unsqueeze(0).unsqueeze(0)

    def get_image(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        ret = self.process_image(frame)
        return ret

    def streaming(self):
        self.is_streaming = True
        while self.is_streaming:
            image = self.get_image()
            if image is not None:
                self.image[:] = image

    def stop_streaming(self):
        self.is_streaming = False
        self.cap.release()
        cv2.destroyAllWindows()
