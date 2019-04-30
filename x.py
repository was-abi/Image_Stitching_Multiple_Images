import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

a = cv2.imread('2.png')
a1=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)

b= cv2.imread('1.png')
b1=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp1,des1 = sift.detectAndCompute(a1,None)
kp2,des2 = sift.detectAndCompute(b1,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
good = []
for m in matches:
    if (m[0].distance<0.5*m[1].distance):
        good.append(m)
        matches=np.asarray(good)

if (len(matches[:,0]) >=4):
    src= np.float32([kp1[m.queryIdx].pt for m in matches[:,0]]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches[:,0]]).reshape(-1,1,2)

    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

else:
    raise AssertionError("Can’t find enough keypoints.")

dst = cv2.warpPerspective(a,H,(b.shape[1] + a.shape[1], b.shape[0]))
plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
plt.show()
plt.figure()
dst[0:a.shape[0], 0:a.shape[1]] = a
cv2.imwrite('output.jpg',dst)
plt.imshow(dst)
plt.show()
