import cv2
import numpy as np
path = r"C:\Users\Public\Pictures\Sample Pictures\Hydrangeas.jpg"
img = cv2.imread(path)
normalizedImg = np.zeros((800, 800))
normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('dst_rt', normalizedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()