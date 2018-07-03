from PIL import Image
import cv2
import numpy as np
from math import *
#图片预处理
class Img_pretreatment(object):

    #图片压缩
    def img_compress(url):
        image = Image.open(url)

        width = image.width
        height = image.height
        rate = 1.0  # 压缩率

        # 根据图像大小设置压缩率
        if width >= 1000 or height >= 1000:
            rate = 0.3
        elif width >= 500 or height >= 500:
            rate = 0.5
        elif width >= 300 or height >= 300:
            rate = 0.9

        width = int(width * rate)  # 新的宽
        height = int(height * rate)  # 新的高

        image.thumbnail((width, height), Image.ANTIALIAS)  # 生成缩略图

        return image

    #图片尺寸调整256*256
    def img_resize(self,image):
        size = 256, 256
        image.resize(size)
        return image

    #图片旋转  cv2图片和角度
    def img_rotate(image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image

        return cv2.warpAffine(image, M, (nW, nH))

    def test(img,degree):
        height, width = img.shape[:2]

        # 旋转后的尺寸
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

        matRotation[0, 2] += (widthNew - width) / 2
        matRotation[1, 2] += (heightNew - height) / 2

        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew))
        return imgRotation
