import tensorflow as tf
from glob import glob
import os
import numpy as np
from lossfunctions import compute_loss, _gram_matrix
from read_images import read_image, preprocess_for_vgg19, preprocess_for_transformnet, postprocess, content_layers, \
    style_layers, num_style_layers, num_content_layers
from transformnet import TransformNet
from vgg import get_model

#pip install -q -U tensorflow-addons

def trainer(content_images, num_images, style_path, batch_size, style_weight, content_weight, variation_weight, saved_model_path):
    Transnet = TransformNet()
    vgg = get_model()
    style_img = read_image(style_path)
    style_img = tf.convert_to_tensor(style_img, tf.float32)
    style_img = tf.expand_dims(style_img, 0)
    style_img = preprocess_for_vgg19(style_img)
    style_outputs = vgg(style_img)
    style_features = [style_layer for style_layer in style_outputs[:num_style_layers]]
    gram_style = [_gram_matrix(tf.convert_to_tensor(style_feature, tf.float32)) for style_feature in style_features]
    epochs = 1
    train_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    #content_images = glob(os.path.join(content_path, "*.jpg"))
    #num_images = len(content_images) - (len(content_images) % batch_size)
    j=0
    for i in range(epochs):
        for p, batch in enumerate([content_images[m:m + batch_size] for m in range(0, num_images, batch_size)]):
            content_image = [read_image(img_path, (256, 256, 3)) for img_path in batch]
            #print(j)
            content_image = np.array(content_image)
            content_image = tf.convert_to_tensor(content_image)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(Transnet.model.trainable_variables)
                # content_img = [read_image(img_path, (256,256,3)) ]

                vgg_content_image = preprocess_for_vgg19(content_image)
                vgg_content_image_outputs = vgg(vgg_content_image)
                Content_image_features = [content_layer for content_layer in
                                          vgg_content_image_outputs[num_style_layers:]]

                transformed_image = preprocess_for_transformnet(content_image)
                transformed_image = Transnet.model(transformed_image)
                transformed_image = postprocess(transformed_image)
                vgg_transformed_image = preprocess_for_vgg19(transformed_image)
                transformed_image_outputs = vgg(vgg_transformed_image)
                Content_transformed_features = [content_layer for content_layer in
                                                transformed_image_outputs[num_style_layers:]]
                Style_transformed_features = [style_layer for style_layer in
                                              transformed_image_outputs[:num_style_layers]]

                loss = compute_loss(Content_image_features, Content_transformed_features, gram_style,
                                    Style_transformed_features, transformed_image, style_weight, content_weight,
                                    variation_weight)
            grads = tape.gradient(loss, Transnet.model.trainable_variables)
            train_optimizer.apply_gradients(zip(grads, Transnet.model.trainable_variables))
            j+=1
            if j % 20 == 0:
                tf.print("total loss is: %f" % (loss))
            if j % 1000 ==0:
                tf.keras.models.save_model(model=Transnet.model, filepath=saved_model_path)
        tf.keras.models.save_model(model=Transnet.model, filepath=saved_model_path)
    return Transnet
            # tf.print("total loss is: %f" % (loss))


dataset_path = '/home/abousis/Ioankont/dataset'
content_images = glob(os.path.join(dataset_path, "*.jpg"))
num_images = len(content_images) - (len(content_images) % 8)

style_path = '/home/abousis/Ioankont/StyleImages/picasso_selfport1907.jpg'
check_path = '/home/abousis/Ioankont/checkpoints'
t=trainer(content_images,num_images,style_path,8,6e1,1e0,5e1,check_path)

#content_path='/content/drive/MyDrive/COCO/2017/train2017_sub-1000/'
#style_path='/content/drive/MyDrive/picturesStyleTransfer/dat/picasso.jpg'

#t=trainer(content_path,style_path,8,4e1,1e0,2e1,'/content/drive/MyDrive/MaxpoolNet/')
