
import cv2

# path
path = 'test.jpg'
src = cv2.imread(path)
window_name = 'Image'
# Using cv2.ROTATE_90_CLOCKWISE rotate
# by 90 degrees clockwise
image = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
