import cv2 as cv 

img = cv.imread("Test_Alphabet/A/0de66b7e-1449-4079-bf2c-935ec261ddcb.rgb_0000.png")
cv.imshow("Display Window", img)
k = cv.waitKey(0)
