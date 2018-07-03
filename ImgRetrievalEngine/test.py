# import cv2
from ImgRetrievalEngine.DBConnect import DBConnectIni
from ImgRetrievalEngine.image_CST_features import ImgCST
from ImgRetrievalEngine.image_ORB_features import OrbDescriptor
# image2 = cv2.imread('F:\\Imagetest\\17.jpg')
# kp1,des1=OrbDescriptor.getORBFeature(image2)
# print(des1)
# import urllib.request as ur
# strHtml = ur.urlopen('http://27.115.11.254:40011/uploads/attachment_images/project/201806/44/IMAGES_1527831198_NwdAat.jpg').read()
# print(strHtml)
import numpy as np
import urllib.request as ur
import cv2


# URL到图片
from ImgRetrievalEngine.image_asift_features import ImgAffineSift

db=DBConnectIni.connection("/PuminTech/database.config")
cur=db.cursor()
print(cur)
db.close()
# def url_to_image(url):
#     # download the image, convert it to a NumPy array, and then read
#     # it into OpenCV format
#     resp = ur.urlopen(url)
#     # bytearray将数据转换成（返回）一个新的字节数组
#     # asarray 复制数据，将结构化数据转换成ndarray
#     image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     # cv2.imdecode()函数将数据解码成Opencv图像格式
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#     # return the image
#     # print(image)
#     return image
#
#
# # initialize the list of image URLs to download
# urls = "http://27.115.11.254:40011/uploads/attachment_images/project/201806/44/IMAGES_1527831198_NwdAat.jpg"
# image = url_to_image(urls)
# kp1,des1=OrbDescriptor.getORBFeature(image)
# print(des1)
# desc=ImgCST.getColorMoments(image)
# print(desc)
# kp,desA=ImgAffineSift.getASiftFeature(image)
# print(desA)
# cv2.imshow("Image", image)
# cv2.waitKey(0)
