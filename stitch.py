# import the necessary packages
from pano_pyims import Stitcher
import argparse
import imutils
import cv2
import numpy as np
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame
"""
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
                help="path to the right image")
ap.add_argument("-s", "--second", required=True,
                help="path to the left image")
args = vars(ap.parse_args())
# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
"""
imageA = cv2.imread("images/1_came.jpg")
imageB = cv2.imread("images/2_came.jpg")
imageA = imutils.resize(imageA, width=800,height=800)
imageB = imutils.resize(imageB, width=800,height=800)
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA,imageB], showMatches=True)
# show the images
#cv2.imshow("Image A", imageA)
#cv2.imshow("Image B", imageB)
#cv2.imshow("Keypoint Matches", vis)
#cv2.imshow("Result", result)
#cv2.waitKey(0)
cv2.imwrite("images/output/"+str(1)+"_out.jpg",result)

imageA = cv2.imread("images/3_came.jpg")
imageB = cv2.imread("images/4_came.jpg")
imageA = imutils.resize(imageA, width=800,height=800)
imageB = imutils.resize(imageB, width=800,height=800)
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA,imageB], showMatches=True)
# show the images
#cv2.imshow("Image A", imageA)
#cv2.imshow("Image B", imageB)
#cv2.imshow("Keypoint Matches", vis)
#cv2.imshow("Result", result)
#cv2.waitKey(0)
cv2.imwrite("images/output/"+str(2)+"_out.jpg",result)

imageA = cv2.imread("images/output/1_out.jpg")
imageB = cv2.imread("images/output/2_out.jpg")
imageA = imutils.resize(imageA, width=800,height=800)
imageB = imutils.resize(imageB, width=800,height=800)
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA,imageB], showMatches=True)
# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
