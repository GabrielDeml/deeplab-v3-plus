import cv2
import numpy as np


imgReal = cv2.imread("8c519ece-0000000_mask.png", 1)

kernel = np.ones((5,5), np.uint8)
imgReal = cv2.dilate(imgReal, kernel, iterations=3)
# lower = np.array(lower, dtype = "uint8")

maskG = cv2.inRange(imgReal, np.array([0, 128, 0]), np.array([0, 128, 0]))
maskR = cv2.inRange(imgReal, np.array([128, 0, 0]), np.array([128, 0, 0]))
maskB = cv2.inRange(imgReal, np.array([0, 0, 128]), np.array([0, 0, 128]))

# calculate moments of binary image
def findCenter(img):
    centers =[]
    im2, contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(imgReal, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(imgReal, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        centers.append((cX, cY))
    return centers

print("Found balls: "+ str(findCenter(maskG)))
print("Found Targets: " + str(findCenter(maskR)))
print("Found hatchpanels: "+ str(findCenter(maskB)))


cv2.imshow("foundBlob", imgReal)
cv2.waitKey(0)
# cv2.imshow("thing", img)

