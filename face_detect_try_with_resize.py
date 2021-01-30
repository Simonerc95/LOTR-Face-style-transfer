import cv2
import os
from os.path import join
from mtcnn import MTCNN
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if not os.path.exists('cropped'):
    os.mkdir('cropped')

for im_name in os.listdir('data'):
    img = cv2.imread(join('data', im_name))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    detected = detector.detect_faces(img)

    # Draw the rectangle around each face
    for face in detected:
        (x, y, w, h) = face['box']
        l = max(w,h)
        img = np.array(img)
        # img = cv2.rectangle(img, (x-int(0.4*l), max(0, y-int(0.4*l))), (x+int(1.2*l), y+int(1.2*l)), (255, 0, 0), 2)

        img_crop = img[max(0, y-int(0.4*l)):y+int(1.2*l), (x-int(0.4*l)):(x+int(1.2*l)), :]
        w, h = img_crop.shape[0], img_crop.shape[1] #I need these to upsize later
        dim = (w, h)
        the_cropped_resized = cv2.resize(img_crop, (256,256))
        cv2.imwrite(join('cropped',f'{im_name[:-4]}_crop.jpg'), the_cropped_resized)

        #just a try 
        the_cropped_upsized = cv2.resize(the_cropped_resized, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(join('cropped',f'{im_name[:-4]}_DEcrop.jpg'), the_cropped_upsized)
        # for (x, y) in face['keypoints'].values():
        #     img = cv2.circle(img, (x,y), radius=1, color=(0, 0, 255), thickness=-1)
    # Display
    cv2.imshow('img', img_crop) # waits until a key is pressed
    cv2.waitKey(0)
    cv2.imshow('resized', the_cropped_upsized)
    cv2.destroyAllWindows() # destroys the window showing image
