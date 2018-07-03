# coding=utf-8
import cv2
import numpy as np


class OrbDescriptor(object):

    def getORBFeature(img):
        # 最大特征点数,需要修改，5000太大。
        orb = cv2.ORB_create(600)
        kp, des = orb.detectAndCompute(img, None)

        return kp,des

        # 蛮力匹配,不做任何筛选。
    def bfdes(des1, des2):
        # 保留最大的特征点数目
        # 找到ORB特征点并计算特征值
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return len(matches)

    # knn筛选匹配点，数据较大的时候建议筛选
    def knndes(des1, des2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
        good = [m for (m, n) in matches if m.distance < 0.75* n.distance]
        return len(good)
        # return good


    def pca(des):
        average = np.mean(des, axis=0)
        m, n = np.shape(des)
        meanRemoved = des - np.tile(average, (m, 1))
        normData = meanRemoved / np.std(des)
        covMat = np.cov(normData.T)
        eigValue, eigVec = np.linalg.eig(covMat)
        eigValInd = np.argsort(-eigValue)
        selectVec = np.matrix(eigVec.T[:32],dtype = np.float32)
        finalData = normData * selectVec.T
        # finalData=np.array(finalData)
        return finalData