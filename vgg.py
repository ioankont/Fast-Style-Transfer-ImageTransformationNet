import tensorflow as tf

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from tensorflow.keras import Model

from read_images import style_layers,content_layers,num_content_layers,num_style_layers


def get_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    #vgg.trainable = False
    #vgg=replace_max_by_average_pooling(vgg)
    #style_outputs = [vgg.get_layer(name).output for name in style_layers]
    #content_outputs = [vgg.get_layer(name).output for name in content_layers]
    total_output_layers = style_layers + content_layers
    #model_outputs = style_outputs + content_outputs
    model_outputs = [vgg.get_layer(layer).output for layer in total_output_layers]
    return Model(vgg.input, model_outputs, trainable=False)
