
import tensorflow as tf
from tensorflow.contrib import slim


class Lenet(object):
    def __init__(self, inputs, scope='lenet', training_flag=True, reuse=False):
        self.scope=scope
        self.inputs=inputs
        if inputs.get_shape()[3] == 3:
            self.inputs = tf.image.rgb_to_grayscale(self.inputs)
        self.training_flag=training_flag
        self.is_training=True
        self.reuse=reuse
        self.create()


    def create(self,is_training=False):

        with tf.variable_scope(self.scope, reuse=self.reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
                    net=self.inputs
                    net = slim.conv2d(net, 32, 1, scope='conv1')
                    self.conv1=net
                    net = slim.max_pool2d(net, 1, stride=2, scope='pool1')
                    self.pool1 = net
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net,64, 1, scope='conv2')
                    self.conv2= net
                    net = slim.max_pool2d(net, 1, stride=2, scope='pool2')
                    self.pool2= net
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, 128, 1, scope='conv3')
                    self.conv3=net
                    net = slim.max_pool2d(net, 1, stride=2, scope='pool3')
                    self.pool3 = net
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net,256, 1, scope='conv4')
                    self.conv4= net
                    net = slim.max_pool2d(net, 1, stride=2, scope='pool4')
                    self.pool4= net
                    net = slim.batch_norm(net)


                    net = slim.conv2d(net, 512, 1, scope='conv5')
                    self.conv5=net
                    net = slim.max_pool2d(net, 1, stride=2, scope='pool5')
                    self.pool5 = net
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net,1024, 1, scope='conv6')
                    self.conv6= net
                    net = slim.max_pool2d(net, 1, stride=2, scope='pool6')
                    self.pool6= net
                    net = slim.batch_norm(net)

                    net = slim.conv2d(net, 1024, 1, scope='conv7')
                    self.conv7=net
                    net = slim.max_pool2d(net, 1, stride=2, scope='pool7')
                    self.pool7 = net
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net,512, 1, scope='conv8')
                    self.conv8= net
                    net = slim.max_pool2d(net, 1, stride=2, scope='pool8')
                    self.pool8= net
                    net = slim.batch_norm(net)


                    net = slim.conv2d(net, 256, 1, scope='conv9')
                    self.conv9=net
                    net = slim.max_pool2d(net, 1, stride=2, scope='pool9')
                    self.pool9 = net
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net,128, 1, scope='conv10')
                    self.conv10= net
                    net = slim.max_pool2d(net, 1, stride=2, scope='pool10')
                    self.pool10= net
                    net = slim.batch_norm(net)


                    net = tf.contrib.layers.flatten(net)
                    net = slim.fully_connected(net, 64, activation_fn=tf.nn.relu, scope='fc3')
                    self.fc3= net
                    #net = slim.dropout(net,0.5, is_training=self.training_flag)

                    
                    ################################################################################
                    #  tf.nn.tanh
                    net = slim.fully_connected(net,16, activation_fn=tf.nn.relu,scope='fc4')
                    self.fc4 = net
                    ### number of outputs is the number of classes such as 3 or 5 or 2.
                    net = slim.fully_connected(net,3, activation_fn=None, scope='fc5')
                    self.fc5 = net
                    self.softmax_output=slim.softmax(net,scope='prediction')

