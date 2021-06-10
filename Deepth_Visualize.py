import cv2
import numpy as np
import matplotlib.pyplot as plt


depth_filename = "raw_depth.npy"
file = np.load(depth_filename)
print(file.shape)

plt.imshow(file)
plt.show()

cv2.imshow('depth', file)
cv2.waitKey(0)

depth = cv2.imread('raw_depth.exr', -1)
cv2.imshow('depth', depth)
cv2.waitKey(0)


