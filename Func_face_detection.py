import cv2
import os
from os.path import join
from mtcnn import MTCNN
import numpy as np

#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_detection(img_path,im_name, _res, tmp_path):

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    h, w = img.shape[:-1]
    if max(h,w) > 1024:
        max_res = 1024 / max(h,w)
        img = cv2.resize(img, (int(max_res*w), int(max_res*h)))
    detector = MTCNN()
    detected = detector.detect_faces(img)

    # Crop a square bounding box around the first face
    try:
        face = detected[0]
        (x, y, w, h) = face['box']
        l = max(w,h)
        scale = 0.75 # take a crop 75% larger than the detected bb
        crop_side = int(l + scale*l)
        img = np.array(img)
        start_y = max(0, y-int((scale/2)*l))
        start_x = max(0, x-int((scale/2)*l))
        img_crop = img[start_y:start_y+crop_side, start_x:start_x+crop_side, :]
    except:
        center = np.array(img.shape[:-1]) // 2
        h, w = img.shape[:-1]
        crop_side = int(min(w, h))
        start_x = max(0, int(center[1] - crop_side/2))
        start_y = max(0, int(center[0] - crop_side/2))
        img_crop = img[int(start_y):int(start_y + crop_side), int(start_x):int(start_x + crop_side)]

    cv2.imwrite(join(tmp_path,f'{im_name}_crop.jpg'), cv2.resize(img_crop, (_res,_res)))

    return join(tmp_path,f'{im_name}_crop.jpg'), (start_y, start_y+crop_side, start_x, start_x+crop_side), img_crop.shape[:2]


