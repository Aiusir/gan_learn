import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random


class Generator():
    def __init__(self,batch_size=128,x_size=(28,28),h_dim = 128):
        self.reuse = False
        self.x_size = x_size
        self.batch_size = batch_size
        self.h_dim = h_dim

    def __call__(self,z,y,name =''):
        inputs = tf.concat([z,y],1)
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('g',reuse = self.reuse):
            # reshape from inputs
            with tf.variable_scope('reshape'):
                outputs = tf.reshape(inputs,[self.batch_size,-1])
            with tf.variable_scope('dense1'):
                outputs = tf.layers.dense(outputs,self.h_dim,activation = tf.nn.relu)
            with tf.variable_scope('dense2'):
                outputs = tf.layers.dense(outputs,self.x_size[0]*self.x_size[1],activation = tf.nn.sigmoid)
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'g')
        return outputs


class Discriminator():
    def __init__(self,batch_size=128,h_dim = 128):
        self.reuse = False
        self.batch_size = batch_size
        self.h_dim = h_dim

    def __call__(self,x,y,name=''):
        inputs = tf.concat([x,y],1)
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('d',reuse = self.reuse):
            # reshape from inputs
            with tf.variable_scope('reshape'):
                outputs = tf.reshape(inputs,[self.batch_size,-1])
            with tf.variable_scope('dense1'):
                outputs_logits = tf.layers.dense(outputs,self.h_dim,activation = tf.nn.relu)
            with tf.variable_scope('dense2'):
                outputs_prob = tf.layers.dense(outputs_logits,1,activation = tf.nn.sigmoid)
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'd')
        return outputs_prob,outputs_logits


class CGANMLP():
    def __init__(self,batch_size=128,x_size = (28,28),h_dim = 128,z_dim = 100):
        self.batch_size = batch_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.x_size = x_size
        self.g = Generator(batch_size=self.batch_size,x_size = self.x_size,h_dim = self.h_dim)
        self.d = Discriminator(batch_size=self.batch_size,h_dim = self.h_dim)
        self.z = tf.random_uniform([self.batch_size,self.z_dim],minval = -1.0,maxval = 1.0)
    
    def loss(self,x_in,y_in):
        '''build model,calculate losses.
        Args:
            x_in:4-D Tensor of shape '[batch,height,width,channels]'
            y_in:2-D Tensor of shape '[batch,label_num]'
        Returns:
            dict of each models loss.
        '''
        generated = self.g(self.z,y_in)
        d_real,d_logits_real = self.d(x_in,y_in,name = 'd')
        d_fake,d_logits_fake = self.d(generated,y_in,name = 'g')

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels = tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels = tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_fake+d_loss_real
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels = tf.ones_like(d_logits_fake)))
        return {self.g:g_loss,self.d:d_loss}

    def train(self,losses,learning_rate = 0.0002):
        '''
        Args:
            losses dict
        Returns:
            train op.
        '''
        d_opt_op = tf.train.AdamOptimizer(learning_rate).minimize(losses[self.d],var_list = self.d.variables)
        g_opt_op = tf.train.AdamOptimizer(learning_rate).minimize(losses[self.g],var_list = self.g.variables)

        with tf.control_dependencies([d_opt_op,g_opt_op]):
            return tf.no_op(name = 'train')

    def sample_images(self,inputs_z,x_size=(28,28),row=8,col =8,inputs = None):
        if inputs is None:
            inputs = self.z
        imgs = self.g(inputs,inputs_z)
        imgs = tf.reshape(imgs,[self.batch_size,x_size[0],x_size[1],1])
        imgs = tf.image.convert_image_dtype(imgs,tf.uint8)
        imgs = [img for img in tf.split(imgs,self.batch_size,axis = 0)]

        rows = []
        for i in range(row):
            rows.append(tf.concat(imgs[col*i+0:col*i+col],2))
        img = tf.concat(rows,1)
        return tf.image.encode_jpeg(tf.squeeze(img,[0]))

def train():
    batch_size = 128
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

    cgan = CGANMLP()
    x_data = tf.placeholder(tf.float32,[batch_size,28*28],name = 'x_data')
    y_data = tf.placeholder(tf.float32,[batch_size,10],name = 'y_data')

    losses = cgan.loss(x_data,y_data)
    train_op = cgan.train(losses,learning_rate = 0.0001)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for step in range(1000000):
            x_in,y_in = mnist.train.next_batch(batch_size)
            _,g_loss_value,d_loss_value = sess.run([train_op,losses[cgan.g],losses[cgan.d]],feed_dict = {x_data:x_in,y_data:y_in})

            if step%1000==0:
                print step,g_loss_value,d_loss_value

            if step%10000==0:
                label = random.randint(0,9)
                y_input = tf.one_hot(label,depth = 10)
                y_titl = tf.tile(y_input,[batch_size])
                y_input = tf.reshape(y_titl,[batch_size,10])
                images = cgan.sample_images(y_input)
                generated = sess.run(images)
                with open('train%d_%d.jpg'%(step,label),'wb')as f:
                    f.write(generated)
        saver.save(sess,os.path.join('./model','model.ckpt'))

if __name__ == '__main__':
    train()









    
