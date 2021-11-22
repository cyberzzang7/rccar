import cv2
import numpy as np


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 0, 255)

    return canny
def reg_of_int(img):

    # Image Size
    height = img.shape[0]

    # Set Region
    region = np.array([
        [(0, height), (1900, height), (500, 0)]
    ])
    # Apply Mask to the Image
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, region, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(0, height), (1900, height), (500, 0)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = reg_of_int(canny_image)

    cv2.imshow("qwer", cropped_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()