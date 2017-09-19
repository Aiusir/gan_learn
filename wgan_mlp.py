import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import os


class Generator:
    def __init__(self,batch_size=128,x_size=(28,28),h_dim = 128):
        self.reuse = False
        self.x_size = x_size
        self.batch_size = batch_size
        self.h_dim = h_dim

    def __call__(self,z,training=False,name =''):
        inputs = tf.convert_to_tensor(z)
        with tf.variable_scope('g',reuse = self.reuse):
            # reshape from inputs
            with tf.variable_scope('reshape'):
                outputs = tf.reshape(inputs,[self.batch_size,-1])
            with tf.variable_scope('dense1'):
                outputs = tf.layers.dense(outputs,self.h_dim,activation = tf.nn.relu,kernel_initializer = tf.random_normal_initializer(0,0.02),trainable = training)
            with tf.variable_scope('dense2'):
                outputs = tf.layers.dense(outputs,self.x_size[0]*self.x_size[1],activation = tf.nn.sigmoid,kernel_initializer = tf.random_normal_initializer(0,0.02),trainable = training)
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'g')
        return outputs


class Discriminator:
    def __init__(self,batch_size=128,h_dim = 128):
        self.reuse = False
        self.batch_size = batch_size
        self.h_dim = h_dim

    def __call__(self,x,training=False,name=''):
        inputs = tf.convert_to_tensor(x)
        with tf.name_scope('d'+name), tf.variable_scope('d',reuse = self.reuse):
            # reshape from inputs
            with tf.variable_scope('reshape'):
                outputs = tf.reshape(inputs,[self.batch_size,-1])
            with tf.variable_scope('dense1'):
                outputs = tf.layers.dense(outputs,self.h_dim,activation = tf.nn.relu,kernel_initializer = tf.random_normal_initializer(0,0.02),trainable = training)
            with tf.variable_scope('dense2'):
                outputs_logits = tf.layers.dense(outputs,1,kernel_initializer = tf.random_normal_initializer(0,0.02),trainable = training)
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'd')
        return outputs_logits


class WGANMLP:
    def __init__(self,batch_size=128,x_size = (28,28),h_dim = 128,z_dim = 100):
        self.batch_size = batch_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.x_size = x_size
        self.g = Generator(batch_size=self.batch_size,x_size = self.x_size,h_dim = self.h_dim)
        self.d = Discriminator(batch_size=self.batch_size,h_dim = self.h_dim)
        self.z = tf.random_uniform([self.batch_size,self.z_dim],minval = -1.0,maxval = 1.0)
    
    def loss(self,x_in):
        """build model,calculate losses.
        Args:
            x_in:4-D Tensor of shape '[batch,height,width,channels]'
            y_in:2-D Tensor of shape '[batch,label_num]'
        Returns:
            dict of each models loss.
        """
        generated = self.g(self.z,training = True)
        d_logits_real = self.d(x_in,training = True,name = 'd')
        d_logits_fake = self.d(generated,training = True,name = 'g')

        d_loss = -tf.reduce_mean(d_logits_real)+tf.reduce_mean(d_logits_fake)
        g_loss = -tf.reduce_mean(d_logits_fake)
        return {self.g:g_loss,self.d:d_loss}

    def train(self,losses,learning_rate = 5e-5):
        """
        Args:
            losses dict
        Returns:
            train op.
        """
        d_opt_op = tf.train.RMSPropOptimizer(learning_rate).minimize(losses[self.d],var_list = self.d.variables)
        clip_d_op = [var.assign(tf.clip_by_value(var,-0.01,0.01)) for var in self.d.variables]
        g_opt_op = tf.train.RMSPropOptimizer(learning_rate).minimize(losses[self.g],var_list = self.g.variables)
        with tf.control_dependencies([d_opt_op]):
            train_d_op = tf.tuple(clip_d_op)

        with tf.control_dependencies([g_opt_op]):
            train_g_op =  tf.no_op(name = 'train_g_op')
        
        return train_d_op,train_g_op

    def sample_images(self,x_size=(28,28),row=8,col =8,inputs = None):
        if inputs is None:
            inputs = self.z
        imgs = self.g(inputs,training = True)
        imgs = tf.reshape(imgs,[self.batch_size,x_size[0],x_size[1],1])
        imgs = tf.image.convert_image_dtype(imgs,tf.uint8)
        imgs = [img for img in tf.split(imgs,self.batch_size,axis = 0)]

        rows = []
        for i in range(row):
            rows.append(tf.concat(imgs[col*i+0:col*i+col],2))
        img = tf.concat(rows,1)
        return tf.image.encode_jpeg(tf.squeeze(img,[0]))

def mnisttrain():
    batch_size = 128
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

    wgan = WGANMLP()
    x_data = tf.placeholder(tf.float32,[batch_size,28*28],name = 'x_data')

    losses = wgan.loss(x_data)
    train_d_op,train_g_op = wgan.train(losses,learning_rate = 5e-5)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for step in range(500000):
            x_in,_ = mnist.train.next_batch(batch_size)
            for _ in range(5):
                sess.run(train_d_op,feed_dict = {x_data:x_in})
            _,g_loss_value,d_loss_value = sess.run([train_g_op,losses[wgan.g],losses[wgan.d]],feed_dict = {x_data:x_in})

            if step%1000==0:
                print step,g_loss_value,d_loss_value

            if step%10000==0:
                images = wgan.sample_images()
                generated = sess.run(images)
                with open('train%d.jpg'%(step),'wb')as f:
                    f.write(generated)
        saver.save(sess,os.path.join('./model','model.ckpt'))

def mnisttest():
    batch_size = 128

    with tf.Session() as sess:
        wgan = WGANMLP()
        images = wgan.sample_images()

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,os.path.join('./model','model.ckpt'))

        generated = sess.run(images)
        with open('train.jpg','wb')as f:
            f.write(generated)

if __name__ == '__main__':
    mnisttrain()









    
