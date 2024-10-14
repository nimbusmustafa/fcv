import cv2

img = cv2.imread('wallpaper.png')

# this is pixel of 0th row and 0th column
cv2.imshow("sid the sloth", img)
print(img[0][0])
(b_channel, g_channel, r_channel) = cv2.split(img)
cv2.imshow('blue channel',b_channel)
cv2.imshow('green channel',g_channel)
cv2.imshow('red channel',r_channel)
cv2.waitKey(0)


cv2.destroyAllWindows()

