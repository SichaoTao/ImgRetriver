import cv2
import numpy
import numpy as np
from ImgRetrievalEngine.lshash import LSHash


class ImgCST(object):
    # 颜色矩阵特征
    # Compute low order moments(1,2,3)
    def getColorMoments(img):
        hsv = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_BGR2HSV)
        # Split the channels - h,s,v
        h, s, v = cv2.split(hsv)
        # Initialize the color feature
        color_feature = []
        color_feature_ava = []
        # N = h.shape[0] * h.shape[1]
        # The first central moment - average
        h_mean = np.mean(h)  # np.sum(h)/float(N)
        s_mean = np.mean(s)  # np.sum(s)/float(N)
        v_mean = np.mean(v)  # np.sum(v)/float(N)
        color_feature.extend([h_mean, s_mean, v_mean])
        # The second central moment - standard deviation
        h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
        s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
        v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
        color_feature.extend([h_std, s_std, v_std])
        # The third central moment - the third root of the skewness
        h_skewness = np.mean(abs(h - h.mean()) ** 3)
        s_skewness = np.mean(abs(s - s.mean()) ** 3)
        v_skewness = np.mean(abs(v - v.mean()) ** 3)
        h_thirdMoment = h_skewness ** (1. / 3)
        s_thirdMoment = s_skewness ** (1. / 3)
        v_thirdMoment = v_skewness ** (1. / 3)
        color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
        # color_num = ''.join(str(v) for v in color_feature)



        color_feature_ava.append(ImgCST.ava(color_feature[0],[1],[2]).tolist())
        color_feature_ava.append(ImgCST.ava(color_feature[3],[4],[5]))
        color_feature_ava.append(ImgCST.ava(color_feature[6], [7], [8]))
        # color_feature = ImgCST.MaxMinNormalization(color_feature)

        # print(numpy.array(color_feature_ava))

        return color_feature

    def ava(k1,k2,k3):
        return (k1+k2+k3)/3
    # 两个颜色矩距离
    def diastanceColorFeature(des1, des2):
        # lsh = LSHash(6, 9)
        # lsh.index(des1)
        # distance = lsh.query(des2)
        #
        # dis = distance[0]
        des1 = np.array(des1)
        des2 = np.array(des2)
        dis = np.linalg.norm(des1 - des2)
        return dis

    def MaxMinNormalization(d):
        k=[]
        max=np.max(d)
        min = np.min(d)
        for x in d:
            x = (x - min) / (max - min)
            k.append(x)
        return k