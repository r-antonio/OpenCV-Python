import tensorflow as tf
import tflearn
import numpy as np
import math


def input_preprocessing(x):
    img_prep = tflearn.ImagePreprocessing()
    # Zero Center (With mean computed over the whole dataset)
    img_prep.add_featurewise_zero_center()
    # STD Normalization (With std computed over the whole dataset)
    img_prep.add_featurewise_stdnorm()

    img_aug = tflearn.ImageAugmentation()
    # Random flip an image
    img_aug.add_random_flip_leftright()

    x = tflearn.input_data(shape=None,name='input',placeholder=x,data_preprocessing=img_prep,data_augmentation=img_aug)
    return x

class TFLearnModel:
    def __init__(self,graph,id,layers=[]):
        self.graph=graph
        self.id=id
        self.layers=layers

def simple_feed(classes,x):
        x=input_preprocessing(x)
        x = tflearn.fully_connected(x, classes, activation='softmax',restore=False)
        return TFLearnModel(x,"simple_feed")
def two_layers_feed(classes,x):
        x=input_preprocessing(x)
        x = tflearn.fully_connected(x, 128, activation='relu')
        x = tflearn.fully_connected(x, classes, activation='softmax',restore=False)
        return TFLearnModel(x,"two_layers_feed")

def conv_simple(classes,x):

    x=input_preprocessing(x)
    #image=tf.reshape(x, [-1, image_size[0],image_size[1], image_size[2]])
    conv_filters=[32,64,128,256]
    filter_size=3
    i=0
    layer_names=[]
    for filters in conv_filters:
            x = tflearn.conv_2d(x, filters, filter_size, activation='relu',name='conv_%d_1' % i)
            layer_names.append(x.name)
            x = tflearn.conv_2d(x, filters, filter_size, activation='relu',name='conv_%d_2' % i)
            layer_names.append(x.name)
            x = tflearn.max_pool_2d(x, 2, strides=2,name="mp_%d" %i)
            x = tflearn.batch_normalization(x,name="bn%d" % i)
            i=i+1
    fc_hidden=[512]
    i=0
    for h in fc_hidden:
        x = tflearn.fully_connected(x, h, activation='elu',name='fc_%d' % i)
        # x = tflearn.batch_normalization(x)
        i=i+1
    x = tflearn.fully_connected(x, classes, activation='softmax')
    return TFLearnModel(x,"conv_simple",layer_names)



#from __future__ import division, print_function, absolute_import



def all_convolutional(classes,x):
    #https://arxiv.org/pdf/1412.6806.pdf
    image_size=32
    x_size=image_size
    x=input_preprocessing(x)
    conv_filters=[96,96,96,192,192,192]
    conv_strides=[1,1,2,1,1,2]
    filter_size=3
    assert(len(conv_filters)==len(conv_strides))
    for i in range(len(conv_filters)):
            filters=conv_filters[i]
            strides=conv_strides[i]
            x = tflearn.conv_2d(x, filters, filter_size,strides=strides, activation='relu',name='conv_%d' % i)
            x = tflearn.batch_normalization(x)
            x_size=x_size/strides

    x = tflearn.conv_2d(x, 192, 1,scope='conv_final')
    x = tflearn.conv_2d(x, classes, 1,scope='conv_before_gap')
    x = tflearn.avg_pool_2d(x, math.floor(x_size))
    x=tflearn.flatten(x)
    x = tflearn.activation (x, activation='softmax', name='softmax')
    return TFLearnModel(x,"all_conv")

def vgg16(classes,x):
    #https://github.com/tflearn/models
    #code: https://github.com/tflearn/models/blob/master/images/vgg16.py
    #model: https://www.dropbox.com/s/9li9mi4105jf45v/vgg16.tflearn?dl=0
    x=input_preprocessing(x)


    # size=tf.constant(np.array([224,224], dtype=np.int32))
    # x=tf.image.resize_images(x, size)

    # x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_1')
    # x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    # x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')
    #
    # x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    # x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    # x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')
    #
    # x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    # x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    # x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    # x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')
    #
    # x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    # x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    # x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    # x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')
    #
    # x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    # x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    # x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    # x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')
    #
    # x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    # x = tflearn.dropout(x, 0.5, name='dropout1')
    #
    # x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    # x = tflearn.dropout(x, 0.5, name='dropout2')
    #
    # x = tflearn.fully_connected(x, classes, activation='softmax', scope='fc8', restore=False)


    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu')
    x = tflearn.conv_2d(x, 128, 3, activation='relu')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu')
    x = tflearn.conv_2d(x, 256, 3, activation='relu')
    x = tflearn.conv_2d(x, 256, 3, activation='relu')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu')
    x = tflearn.conv_2d(x, 512, 3, activation='relu')
    x = tflearn.conv_2d(x, 512, 3, activation='relu')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu')
    x = tflearn.conv_2d(x, 512, 3, activation='relu')
    x = tflearn.conv_2d(x, 512, 3, activation='relu')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, classes, activation='softmax',restore=False)

    return TFLearnModel(x,"vgg16")

def inception(classes,x):
	# https://github.com/tflearn/tflearn/blob/master/examples/images/googlenet.py

	from tflearn.layers.core import input_data, dropout, fully_connected
	from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
	from tflearn.layers.normalization import local_response_normalization
	from tflearn.layers.merge_ops import merge
	from tflearn.layers.estimator import regression

	network=input_preprocessing(x)
	conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name = 'conv1_7_7_s2')
	pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
	pool1_3_3 = local_response_normalization(pool1_3_3)
	conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
	conv2_3_3 = conv_2d(conv2_3_3_reduce, 192,3, activation='relu', name='conv2_3_3')
	conv2_3_3 = local_response_normalization(conv2_3_3)
	pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
	inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
	inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
	inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
	inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
	inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
	inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
	inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

	# merge the inception_3a__
	inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

	inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
	inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
	inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
	inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
	inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
	inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
	inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')
	
	#merge the inception_3b_*
	inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

	pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
	inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
	inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
	inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
	inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
	inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
	inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

	inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')


	inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
	inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
	inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
	inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')

	inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
	inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')

	inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')


	inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',name='inception_4c_1_1')
	inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
	inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
	inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
	inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')

	inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
	inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')

	inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

	inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
	inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
	inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
	inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
	inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
	inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
	inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')

	inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

	inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
	inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
	inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
	inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
	inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
	inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
	inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')


	inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],axis=3, mode='concat')

	pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')


	inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
	inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
	inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
	inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
	inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
	inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
	inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1,activation='relu', name='inception_5a_pool_1_1')

	inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,mode='concat')


	inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1,activation='relu', name='inception_5b_1_1')
	inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
	inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', name='inception_5b_3_3')
	inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
	inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', name='inception_5b_5_5' )
	inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
	inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
	inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')

	pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
	pool5_7_7 = dropout(pool5_7_7, 0.4)
	loss = fully_connected(pool5_7_7, classes,activation='softmax')
	# network = regression(loss, optimizer='momentum',
        #             loss='categorical_crossentropy',
        #             learning_rate=0.001)
	return TFLearnModel(loss,'inception-v3')

