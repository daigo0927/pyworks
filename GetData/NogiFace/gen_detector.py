#!/usr/bin/python
#! -*- coding: utf-8 -*-

import os
import sys
import glob
import dlib
from skimage import io

# 入力に画像ディレクトリパスを入れなかった時の警告
if len(sys.argv) != 2:
    print(
        "Give the path to the examples/faces directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train_object_detector.py ../examples/faces")
    exit()
faces_folder = sys.argv[1]

# 学習時のパラメータ
options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5
options.num_threads = 4
options.be_verbose = True

# 読みこむファイルパス
training_xml_path = os.path.join(faces_folder, "training.xml")
testing_xml_path = os.path.join(faces_folder, "testing.xml")

# 学習
dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)

# 精度表示
print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))

print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))
