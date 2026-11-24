import cv2
import numpy as np


def prepare_square_input(frame, size=224):
    height, width = frame.shape[:2]

    crop_size = min(height, width)

    x_offset = (width - crop_size) // 2
    y_offset = (height - crop_size) // 2

    cropped = frame[
        y_offset:y_offset + crop_size,
        x_offset:x_offset + crop_size
    ]

    resized = cv2.resize(cropped, (size, size))

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    rgb = np.ascontiguousarray(rgb)

    return rgb, x_offset, y_offset, crop_size