#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os
import sys
import dlib
import numpy as np
import cv2

# 読みこむ画像
img = cv2.imread(sys.argv[1],1)
out = img.copy()

# 検出
detector = dlib.simple_object_detector("detector.svm")
dets = detector(img)

# 四角形描写
print(str(len(dets)))
for d in dets:
    cv2.rectangle(out, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)

cv2.imshow('image',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
