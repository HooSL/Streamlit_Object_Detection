import streamlit as st
import pandas as pd
import numpy as np
from seaborn.axisgrid import pairplot
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
import os
import pathlib

import zipfile

import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import time


def run_image():
    PATH_TO_LABELS = 'C:\\Users\\5-1\\Documents\\TensorFlow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)

    def download_model(model_name, model_date):
        base_url = 'http://download.tensorflow.org/models/object_detection/tf2/' #경로는 변하지 않음
        model_file = model_name + '.tar.gz'
        model_dir = tf.keras.utils.get_file(fname=model_name,
                                            origin=base_url + model_date + '/' + model_file,
                                            untar=True)
        return str(model_dir)
    #모델 날짜와 모델 이름만 바뀜
    MODEL_DATE = '20200711'
    MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
    PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

    def load_model(model_dir):
        model_full_dir = model_dir + "/saved_model"

        # Load saved model and build the detection function
        detection_model = tf.saved_model.load(model_full_dir)
        return detection_model

    detection_model = load_model(PATH_TO_MODEL_DIR)

    st.subheader('이미지를 업로드 해주세요.')
    image_file = st.file_uploader('이미지를 업로드 하세요',type='jpg')
        
    PATH_TO_IMAGE_DIR = pathlib.Path(image_file)
    IMAGE_PATHS = list(PATH_TO_IMAGE_DIR.glob('*.jpg'))

    def load_image_into_numpy_array(path):
        print(str(path))
        return cv2.imread(path)

    for image_path in IMAGE_PATHS:

        print('Running inference for {}... '.format(image_path), end='')



        image_np = load_image_into_numpy_array(image_path)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = detection_model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        print(detections)
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        #cv2.imshow(str(image_path),image_np_with_detections)

        st.image(cv2.imshow(image_path),image_np_with_detections)