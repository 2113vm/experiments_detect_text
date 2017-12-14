from text_detection import *
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('1.jpg')

plt.show()
plt.imshow(image)

contours = find_text(image)

print(contours)
