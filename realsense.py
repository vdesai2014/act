import numpy as np
import pyrealsense2 as rs
import cv2
import torch
import threading

class RealsenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.is_streaming = False
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.image = np.ndarray((3, height, width), dtype=np.uint8)
        threading.Thread(target=self.streaming).start()

    def process_image(self, image):
        image = image.transpose((2, 0, 1))  # Convert to CHW
        image = torch.from_numpy(image).float() / 255.0
        return image.unsqueeze(0).unsqueeze(0)

    def get_image(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        image = np.asanyarray(color_frame.get_data())
        ret = self.process_image(image)
        return ret

    def streaming(self):
        self.pipeline.start(self.config)
        self.is_streaming = True
        while self.is_streaming:
            image = self.get_image()
            if image is not None:
                self.image[:] = image

    def stop_streaming(self):
        self.is_streaming = False
        self.pipeline.stop()

