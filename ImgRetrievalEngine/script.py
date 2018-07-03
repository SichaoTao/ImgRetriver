import os

import cv2

import pickle

from ImgRetrievalEngine.image_CST_features import ImgCST
from ImgRetrievalEngine.image_ORB_features import OrbDescriptor
from ImgRetrievalEngine.image_asift_features import ImgAffineSift
from ImgRetrievalEngine.image_sift_features import SIFT_Feature
import pymysql

db = pymysql.connect("116.62.203.48", "db_shangda", "db_shdx_pwd", "test", charset="utf8")
    # 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()

g = os.walk("F:\\paper")
    # dict = {}
i = 0
for path, dd, filelist in g:
    for filename in filelist:
        url = os.path.join(path, filename)
            # image = Image.open(url)
        image = cv2.imread(url)
        color_feature = ImgCST.getColorMoments(image)
        color_feature = pickle.dumps(color_feature, pickle.HIGHEST_PROTOCOL)

        kp1, orb_feature = OrbDescriptor.getORBFeature(image)
        orb_feature = pickle.dumps(orb_feature, pickle.HIGHEST_PROTOCOL)

        kp2, asift_feture = ImgAffineSift.getASiftFeature(image)

        asift_feture1 = SIFT_Feature.svd(asift_feture)

        asift_feture = pickle.dumps(asift_feture1, pickle.HIGHEST_PROTOCOL)
        print(url, '=====', i)
        i+=1
        sql = "insert into tb_imgretrieval(uuid,color_features,orb_features,asift_features) VALUES(%s,%s,%s,%s)"
        try:
            cursor.execute(sql, (url, color_feature, orb_feature, asift_feture))
            db.commit()

        except Exception as e:
            db.rollback()
            print(e)
    db.close()