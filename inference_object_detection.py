import io
import os
import scipy.misc
import numpy as np
import six
import time
import cv2
from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils


def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Load the COCO Label Map
category_index = {
    1: {'id': 1, 'name': 'apple'},
    2: {'id': 2, 'name': 'banana'},
    3: {'id': 3, 'name': 'orange'}
}
import time
start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load('/home/neeleshkumaryadav/Documents/Tensorflow/fruits/exported-models/ssd_mobilenet_v1_fpn/saved_model/')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')

image_dir = '/home/neeleshkumaryadav/Documents/Tensorflow/fruits/test_images'

elapsed = []
for j, filename in enumerate(os.listdir(image_dir)):
    image_path = os.path.join(image_dir, filename)
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = np.expand_dims(image_np, 0)
    start_time = time.time()
    detections = detect_fn(input_tensor)
    end_time = time.time()
    elapsed.append(end_time - start_time)

    #plt.rcParams['figure.figsize'] = [42, 21]
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.50,
        agnostic_mode=False)
    print("detection boxes:", detections['detection_boxes'][0].numpy())
    print("detection boxes:", detections['detection_classes'][0].numpy().astype(np.int32))
    print("detection_score:", detections['detection_scores'][0].numpy())
    cv2.imwrite("/home/neeleshkumaryadav/Documents/Tensorflow/fruits/output/"+str(j)+".jpg",image_np_with_detections)

mean_elapsed = sum(elapsed) / float(len(elapsed))
print('Elapsed time: ' + str(mean_elapsed) + ' second per image')
#return detections['detection_boxes'][0].numpy()), detections['detection_classes'][0].numpy().astype(np.int32), detections['detection_scores'][0].numpy()
