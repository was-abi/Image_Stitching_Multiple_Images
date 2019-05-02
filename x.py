import cv2
import numpy as np
import matplotlib.pyplot as plt



img_ = cv2.imread('images/2.jpg')
img1 = cv2.cvtColor(img_ , cv2.COLOR_BGR2GRAY)


img = cv2.imread('images/1.jpg')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m in matches:
    if m[0].distance < 0.5*m[1].distance:
        good.append(m)
matches = np.asarray(good)

if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
#print H
else:
    raise AssertionError("Can’t find enough keypoints.")

dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))


#plot cha kaam
plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
plt.show()
plt.figure()
dst[0:img.shape[0], 0:img.shape[1]] = img
cv2.imwrite('4.jpg',dst)
plt.imshow(dst)
plt.show()


img_ = cv2.imread('images/3_cam.jpg')
img1 = cv2.cvtColor(img_ , cv2.COLOR_BGR2GRAY)
img1=cv2.resize(img1,(100,100))

img = cv2.imread('4.jpg')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img2=cv2.resize(img2,(100,100))

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m in matches:
    if m[0].distance < 0.5*m[1].distance:
        good.append(m)
matches = np.asarray(good)

if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
#print H
else:
    raise AssertionError("Can’t find enough keypoints.")

dst1 = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))


plt.subplot(122),plt.imshow(dst1),plt.title('Warped Image')
plt.show()
plt.figure()
dst1[0:img.shape[0], 0:img.shape[1]] = img
cv2.imwrite('5.jpg',dst1)
plt.imshow(dst1)
plt.show()
