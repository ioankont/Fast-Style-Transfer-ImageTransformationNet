import tensorflow as tf
import os
import numpy as np
from skimage import io,transform
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def read_image(img_path, img_size=False):
  img = load_img(img_path)
  img = img_to_array(img, dtype=np.float32)
  # We need to broadcast the image array such that it has a batch dimension 
  if img_size != False:
    img = resize_img(img, img_size)
  return img
def resize_img(img, size):
  if len(size) == 2:
    size += (3,)
  return transform.resize(img, size, preserve_range=True)



def preprocess_for_vgg19(img):
  #img=tf.convert_to_tensor(img)
 
  # We need to broadcast the image array such that it has a batch dimension 
  img = tf.keras.applications.vgg19.preprocess_input(img)
  img = tf.cast(img, tf.float32)

  return img

def preprocess_for_transformnet(img):
    return img / 255.0
def postprocess(img):
    return tf.clip_by_value(img, 0.0, 255.0)
    
content_layers = ['block4_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
