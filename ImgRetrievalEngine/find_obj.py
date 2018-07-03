from __future__ import print_function

import numpy as np
import cv2 as cv
from ImgRetrievalEngine.common import anorm, getsize

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6


def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv.xfeatures2d.SIFT_create(80)
        norm = cv.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv.xfeatures2d.SURF_create(800)
        norm = cv.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv.ORB_create(400)
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv.AKAZE_create()
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv.BRISK_create()
        norm = cv.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv.BFMatcher(norm)
    return detector, matcher

#匹配
def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv.circle(vis, (x1, y1), 2, col, -1)
            cv.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv.line(vis, (x1, y1), (x2, y2), green)

    cv.imshow(win, vis)

    def onmouse(event, x, y, flags, param):
        cur_vis = vis
        if flags & cv.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(np.array(p1) - (x, y)) < r) | (anorm(np.array(p2) - (x, y)) < r)
            idxs = np.where(m)[0]

            kp1s, kp2s = [], []
            for i in idxs:
                (x1, y1), (x2, y2) = p1[i], p2[i]
                col = (red, green)[status[i][0]]
                cv.line(cur_vis, (x1, y1), (x2, y2), col)
                kp1, kp2 = kp_pairs[i]
                kp1s.append(kp1)
                kp2s.append(kp2)
            cur_vis = cv.drawKeypoints(cur_vis, kp1s, None, flags=4, color=kp_color)
            cur_vis[:,w1:] = cv.drawKeypoints(cur_vis[:,w1:], kp2s, None, flags=4, color=kp_color)

        cv.imshow(win, cur_vis)
    cv.setMouseCallback(win, onmouse)
    return vis

