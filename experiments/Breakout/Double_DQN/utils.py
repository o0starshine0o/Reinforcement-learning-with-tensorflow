import cv2
import numpy as np

from config import (INPUT_SHAPE)


# This function can resize to any shape, but was built to resize to 84x84
def process_frame(frame, shape=INPUT_SHAPE):
    """Preprocesses a 210x160x3 frame to 84x84x1 grayscale
    Arguments:
        frame: The frame to process.  Must have values ranging from 0-255
        shape: frame shape
    Returns:
        The processed frame
    """
    # (210, 160, 3)
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
    # (210, 160)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # (160, 160), 不明白这里为什么要用34来裁剪图片
    frame = frame[34:34 + 160, :160]  # crop image
    # (84, 84)
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    # (84, 84, 1)
    frame = frame.reshape((*shape, 1))

    return frame
