import cv2
import numpy
from PIL import Image
from numpy import cov
from scipy import linalg
from numpy import *


class SIFT_Feature(object):

    def getSIFTFeature(image):
        img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create(100)
        kp1, des = sift.detectAndCompute(gray, None)
        #destolist=des.tolist()
        return des

    def getSURFFeature(image):
        img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d.SURF_create(2000)
        kp1, des = surf.detectAndCompute(gray, None)
        #destolist=des.tolist()
        return des

    def siftFeatureMatching(des1,des2):
        # FLANN 是快速最近邻搜索包(Fast_Library_for_Approximate_Nearest_Neighbors)的简称
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2,)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        return len(good)

    def svd(des):
        image2 = Image.open('./media/svd_ sample.jpg')
        des2 = SIFT_Feature.getSIFTFeature(image2)
        U, S, V = numpy.linalg.svd(des2)
        size_projected = 64
        projector = V[:, :size_projected]
        des_new = des.dot(projector)
        return des_new


    def pca(mat, lenth):
        meanval = mean(mat, axis=0)
        rmmeanMat = mat - meanval
        covMat = cov(rmmeanMat, rowvar=0)
        eigval, eigvec = linalg.eig(mat(covMat))
        tfMat = eigvec[0:lenth, :]
        finalData = rmmeanMat * tfMat
        recoMat = finalData * tfMat.T + meanval
        return finalData