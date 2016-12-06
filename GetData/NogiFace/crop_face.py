#!/usr/bin/env python

#
#  crop face from all picture in specific directory in every second
#
#  usage: ./crop_face_always.py origin_directory dist_directory
#
 
import cv2
import math
import numpy as np
import os
import sys
import glob
import time
import shutil
import dlib

def cut_face(origin_path, dist_path) :
    # 各画像の処理    
    img_path_list = glob.glob(origin_path + "/*")
    detector = dlib.simple_object_detector("detector.svm")
    for img_path in img_path_list:  
        print(img_path)
        # ファイル名解析
        base_name = os.path.basename(img_path)
        name,ext = os.path.splitext(base_name)
        if (ext != '.jpg') and (ext != '.jpeg') and (ext != '.png'):
            print("not a picture")
            continue

        img_src = cv2.imread(img_path, 1)
                                       
        #顔判定
        dets = detector(img_src)
    
        # 顔があった場合
        if len(dets) > 0:
            i = 0
            for d in dets:
                face = img_src[d.top():d.bottom(), d.left():d.right()]
                file_name = dist_path + name + "_" + str(i) + ext
                cv2.imwrite(file_name, face )
                i += 1 
        else:
            print("not find any faces")
        shutil.move(img_path, origin_path + "/../finished/")

if __name__ == '__main__':
     
    # 入力の読み込み 
    if len(sys.argv) < 2 :
       exit() 
    else :
       origin_path = sys.argv[1] 

    if len(sys.argv) < 3 :
        dist_path = os.path.dirname(os.path.abspath(__file__)) + "/face/"
    else :
        dist_path = sys.argv[2]

    # 毎秒画像を処理
    while(1):
        cut_face(origin_path, dist_path)
        time.sleep(1)
