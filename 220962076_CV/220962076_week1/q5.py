import cv2

image = cv2.imread('test.jpg')
cv2.imshow('Original Image', image)

down_width = 300
down_height = 200
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)

up_width = 1400
up_height = 1000
up_points = (up_width, up_height)
resized_up = cv2.resize(image, up_points, interpolation=cv2.INTER_LINEAR)

# Display images
cv2.imshow('Resized Down by defining height and width', resized_down)
cv2.imshow('Resized Up image by defining height and width', resized_up)
cv2.waitKey(0)
cv2.destroyAllWindows()