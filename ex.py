from imagecorruptions import corrupt
import cv2
import numpy as np

image = cv2.imread('sample.png')   #被处理的图片
corrupted_image = corrupt(image, corruption_name='snow', severity=5)
cv2.imshow('Image', image)
cv2.imshow('corrupted_image', corrupted_image)   #显示增强结果
cv2.imwrite('corrupted_image.jpg',corrupted_image)	#保存增强结果
cv2.waitKey(0)  
