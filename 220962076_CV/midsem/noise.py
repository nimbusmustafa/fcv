import cv2

img = cv2.imread("/home/student/Documents/Test_Images/q1.png")
cv2.imshow('original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()