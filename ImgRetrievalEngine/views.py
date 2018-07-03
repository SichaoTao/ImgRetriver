import os

from PIL import Image
from django.shortcuts import render, render_to_response
from django.http import HttpResponse, response, request
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import urllib.request as ur
import cv2
import numpy as np
import pickle
import zlib

from ImgRetrievalEngine.DBConnect import DBConnectIni
from ImgRetrievalEngine.image_CST_features import ImgCST
from ImgRetrievalEngine.image_ORB_features import OrbDescriptor
from ImgRetrievalEngine.image_asift_features import ImgAffineSift
from ImgRetrievalEngine.image_sift_features import SIFT_Feature
from ImgRetrievalEngine.models import TbImgretrieval
import pymysql
import time

def uploadImgs(request):
    state=''
    # if request.method == "POST":
    url=request.POST.get('url')
    uuid=request.POST.get('uuid')
    uuid=123456
    db = DBConnectIni.connection("/PuminTech/database.config")
    cursor = db.cursor()

    try:
        # resp = ur.urlopen(url)
        resp = ur.urlopen('http://27.115.11.254:40011/uploads/attachment_images/project/201806/44/IMAGES_1527831198_NwdAat.jpg')
        image = np.asarray(bytearray(resp.read()), dtype="uint8")

        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        color_feature = ImgCST.getColorMoments(image)
        color_feature = pickle.dumps(color_feature, pickle.HIGHEST_PROTOCOL)

        kp1, orb_feature = OrbDescriptor.getORBFeature(image)
        orb_feature = pickle.dumps(orb_feature, pickle.HIGHEST_PROTOCOL)

        # kp2, asift_feture = ImgAffineSift.getASiftFeature(image)
        # print(asift_feture.shape)
        # asift_feture1 = SIFT_Feature.svd(asift_feture)
        # print(asift_feture1.shape)
        # asift_feture = pickle.dumps(asift_feture1, pickle.HIGHEST_PROTOCOL)
        # zip_asift_desc = zlib.compress(asift_feture, zlib.Z_FINISH)

        sql = "insert into tb_imgretrieval(uuid,color_features,orb_features) VALUES(%s,%s,%s)"
        try:
            cursor.execute(sql, (uuid, color_feature, orb_feature))
            db.commit()

        except Exception as e:
            db.rollback()
        db.close()
        state = '200'
    except Exception as e:
        state = '404'
    return JsonResponse({"msg": state})



def retrievalImgs(request):
    dict = {}
    dict2 = {}
    dict3 = {}
    # if request.method == "POST":
    url=request.POST.get('url')
    returned_qty=request.POST.get('k')
    if returned_qty is None:
       returned_qty = 5

    db = DBConnectIni.connection("/PuminTech/database.config")
    cursor = db.cursor()

    image = cv2.imread("E:\\paper\\21.jpg")
    # image = Image.open("E:\\paper\\20180626133319")
    # # resp = ur.urlopen(url)
    # resp = ur.urlopen('http://27.115.11.254:40011/uploads/attachment_images/project/201806/44/IMAGES_1527831198_NwdAat.jpg')
    # image = np.asarray(bytearray(resp.read()), dtype="uint8")
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    color_feature_url = ImgCST.getColorMoments(image)
    kp,orb_feature_url=OrbDescriptor.getORBFeature(image)
    # kp2,asift_feature_url=ImgAffineSift.getASiftFeature(image)
    # asift_feature_url = SIFT_Feature.svd(asift_feature_url)

    cursor.execute("select uuid,color_features from tb_imgretrieval")

    results = cursor.fetchall()

    # color_features_db = TbImgretrieval.objects.all().values('color_features')
    if results:
       for color_feature_db in results:
           color_feature_db1 = pickle.loads(color_feature_db[1])
           dis=ImgCST.diastanceColorFeature(color_feature_url,color_feature_db1)
           dict[color_feature_db[0]] = dis
    k=len(dict)

    dict = sorted(dict.items(), key=lambda item: item[1])

    if 0<k<=100:
         leng = k * 0.5
         dict=dict[:int(leng)]
    elif 100<k<=1000:
         leng = k*0.05+50
         dict = dict[:int(leng)]
    elif k>1000:
         leng = 100
         dict = dict[:int(leng)]
    else :
         return

    #orb select
    for key in dict:
         cursor.execute("select url,orb_features from tb_imgretrieval where uuid='%s'" % key[0])
         orb_result = cursor.fetchone()
         orb_db=pickle.loads(orb_result[1])
         len1=OrbDescriptor.knndes(orb_db,orb_feature_url)
         dict2[orb_result[0]] = len1

    dict2 = sorted(dict2.items(), key=lambda item: item[1], reverse=True)
    dict2=dict2[:returned_qty]

    for d in dict2:
         if d[1]==0:
             dict3[d[0]] = '0%'
         elif 0 < d[1] < 20:
             dict3[d[0]] = '10%'
         elif d[1] >= 300:
             dict3[d[0]] = '100%'
         elif 100 < d[1] < 300:
             dict3[d[0]] = '90%'
         else:
             dict3[d[0]] = str(d[1]) + '%'

    # ASIFT select
    # for key2 in dict2:
    #     cursor.execute("select id,asift_features from tb_imgretrieval where id='%s'" % key2[0])
    #     asift_result = cursor.fetchone()
    #     asift_db = pickle.loads(asift_result[1])
    #     distance=ImgAffineSift.siftFeatureMatching(asift_feature_url,asift_db)
    #     dict3[asift_result[0]] = distance
    # dict3 = sorted(dict3.items(), key=lambda item: item[1], reverse=True)
    #
    # print(dict3[:5])
    db.close()
    return JsonResponse({"msg": dict3})