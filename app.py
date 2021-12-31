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

from image import run_image


def main():
    st.title('Image Detection API')

    menu = ['Home','Image Detection']
    choice = st.selectbox('Home Menu',menu)

    if choice == 'Home':
        st.write('Welcome')

    elif choice == 'Image Detection':
        run_image()




if __name__ == '__main__':
    main()