import cv2
import numpy as np

# Load the image
image = cv2.imread("/home/lucayu/lss-cfar/yolov8_sam/test.png")
h, w = image.shape[:2]

# Load the segmentation mask points from the file
with open("./yolomask_format.txt") as f:
    segment = [np.array(x.split(), dtype=np.float32).reshape(-1, 2) for x in f.read().strip().splitlines() if len(x)]

# Rescale the segment points to match the image dimensions
for s in segment:
    s[:, 0] *= w
    s[:, 1] *= h

# Fill the shape with cyan color (BGR: 255, 255, 0)
cv2.fillPoly(image, [s.astype(np.int32) for s in segment], (255, 255, 0))

# Save the output image
cv2.imwrite("output_filled_cyan.jpg", image)
