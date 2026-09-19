# -*- coding: utf-8 -*-

import cv2

# Read the original image
img = cv2.imread('Assignment_3.jpg') 

# let's downscale the image using new  width and height
down_width = 400
down_height = 887
down_points = (down_width, down_height)
resized_down = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)

# Convert to graycsale
img_gray = cv2.cvtColor(resized_down, cv2.COLOR_BGR2GRAY)

# Display Resized image
cv2.imshow('Resized', resized_down)
cv2.waitKey(0)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(resized_down, (3,3), 0) 

cv2.imshow('Blurred Image', img_blur)
cv2.waitKey(0)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=7) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=7) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=7) # Combined X and Y Sobel Edge Detection

# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)

cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)

cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=140, threshold2=500) # Canny Edge Detection

# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)


cv2.destroyAllWindows()


# apply binary thresholding
ret, thresh = cv2.threshold(img_gray, 20, 100, cv2.THRESH_BINARY)

# visualize the binary image
cv2.imshow('Binary image', thresh)
cv2.waitKey(0)

cv2.imwrite('image_thres1.jpg', thresh)
cv2.destroyAllWindows()

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                                     
# draw contours on the original image
image_copy = img_gray.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
               
print(contours)

# see the results
cv2.imshow('None approximation', image_copy)

print("Hello")
cv2.waitKey(0)

cv2.imwrite('contours_none_image1.jpg', image_copy)

cv2.destroyAllWindows()
