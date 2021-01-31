import cv2
import os
from os.path import join
from mtcnn import MTCNN
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
# Load the cascade
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_detection(img_path,im_name):
  if not os.path.exists('cropped'):
      os.mkdir('cropped')


 
  img = cv2.imread(img_path)
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
      print('img_crop_shape' , img_crop.shape)
      cv2.imwrite(join('cropped',f'{im_name}_crop.jpg'), cv2.resize(img_crop, (256,256)))

  return x,y
      # for (x, y) in face['keypoints'].values():
      #     img = cv2.circle(img, (x,y), radius=1, color=(0, 0, 255), thickness=-1)
  # Display
  # cv2.imshow('img', img_crop) # waits until a key is pressed
  # cv2.waitKey(0)
  # cv2.destroyAllWindows() # destroys the window showing image


if __name__ == "__main__":
  face_detection()