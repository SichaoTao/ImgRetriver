#!/usr/bin/env python

'''
Affine invariant feature-based image matching sample.

This sample is similar to find_obj.py, but uses the affine transformation
space sampling technique, called ASIFT [1]. While the original implementation
is based on SIFT, you can try to use SURF or ORB detectors instead. Homography RANSAC
is used to reject outliers. Threading is used for faster affine sampling.

[1] http://www.ipol.im/pub/algo/my_affine_sift/

USAGE
  asift.py [--feature=<sift|surf|orb|brisk>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
               to feature name to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# built-in modules
import itertools as it
from multiprocessing.pool import ThreadPool

# local modules
# from common import Timer
from ImgRetrievalEngine.find_obj import init_feature, filter_matches, explore_match

class ImgAffineSift(object):

    def getASiftFeature(image):
        feature_name = 'sift-flann'
        detector, matcher = init_feature(feature_name)
        pool = ThreadPool(processes=cv.getNumberOfCPUs())
        kp, desc = ImgAffineSift.affine_detect(detector, image, pool=pool)
        return kp, desc

    def matchASift(kp1, des1, kp2, des2):
        detector, matcher = init_feature('sift-flann')
        raw_matches = matcher.knnMatch(des1, trainDescriptors=des2, k=2)
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
        return np.sum(status)

    def affine_skew(tilt, phi, img, mask=None):
        '''
        affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

        Ai - is an affine transform matrix from skew_img to img
        '''
        h, w = img.shape[:2]
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
            mask[:] = 255
        A = np.float32([[1, 0, 0], [0, 1, 0]])
        if phi != 0.0:
            phi = np.deg2rad(phi)
            s, c = np.sin(phi), np.cos(phi)
            A = np.float32([[c, -s], [s, c]])
            corners = [[0, 0], [w, 0], [w, h], [0, h]]
            tcorners = np.int32(np.dot(corners, A.T))
            x, y, w, h = cv.boundingRect(tcorners.reshape(1, -1, 2))
            A = np.hstack([A, [[-x], [-y]]])
            img = cv.warpAffine(img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        if tilt != 1.0:
            s = 0.8 * np.sqrt(tilt * tilt - 1)
            img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
            img = cv.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv.INTER_NEAREST)
            A[0] /= tilt
        if phi != 0.0 or tilt != 1.0:
            h, w = img.shape[:2]
            mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
        Ai = cv.invertAffineTransform(A)
        return img, mask, Ai

    def affine_detect(detector, img, mask=None, pool=None):
        '''
        affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

        Apply a set of affine transormations to the image, detect keypoints and
        reproject them into initial image coordinates.
        See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

        ThreadPool object may be passed to speedup the computation.
        '''
        params = [(1.0, 0.0)]
        for t in 2 ** (0.5 * np.arange(1, 6)):
            for phi in np.arange(0, 180, 72.0 / t):
                params.append((t, phi))

        def f(p):
            t, phi = p
            timg, tmask, Ai = ImgAffineSift.affine_skew(t, phi, img)
            keypoints, descrs = detector.detectAndCompute(timg, tmask)
            for kp in keypoints:
                x, y = kp.pt
                kp.pt = tuple(np.dot(Ai, (x, y, 1)))
            if descrs is None:
                descrs = []
            return keypoints, descrs

        keypoints, descrs = [], []
        if pool is None:
            ires = it.imap(f, params)
        else:
            ires = pool.imap(f, params)

        for i, (k, d) in enumerate(ires):
            # print('affine sampling: %d / %d\r' % (i + 1, len(params)), end='')
            keypoints.extend(k)
            descrs.extend(d)
        # print(np.array(descrs).shape)
        return keypoints, np.array(descrs)

    def siftFeatureMatching(kp1,des1,kp2, des2):

        # FLANN 是快速最近邻搜索包(Fast_Library_for_Approximate_Nearest_Neighbors)的简称
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2, )
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        return len(good)

    def siftFeatureMatching(des1, des2):
        # FLANN 是快速最近邻搜索包(Fast_Library_for_Approximate_Nearest_Neighbors)的简称
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2, )
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        return len(good)