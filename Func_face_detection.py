import cv2
import os
from os.path import join
from mtcnn import MTCNN
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_detection(img_path,im_name, _res, tmp_path):

    img = cv2.imread(img_path)
    detector = MTCNN()
    detected = detector.detect_faces(img)

    # Crop a square bounding box around the first face
    face = detected[0]
    (x, y, w, h) = face['box']
    l = max(w,h)
    scale = 0.4 # take a crop 40% larger than the detected bb
    crop_side = int(l + scale*l)
    img = np.array(img)
    start_y = max(0, y-int((scale/2)*l))
    start_x = max(0, x-int((scale/2)*l))
    img_crop = img[start_y:start_y+crop_side, start_x:start_x+crop_side, :]
    print('img_crop_shape' , img_crop.shape)
    cv2.imwrite(join(tmp_path,f'{im_name}_crop.jpg'), cv2.resize(img_crop, (_res,_res)))

    return join(tmp_path,f'{im_name}_crop.jpg'), (start_y, start_y+crop_side, start_x, start_x+crop_side), img_crop.shape[:2]


