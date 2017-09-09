import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import os


class Generator:
    def __init__(self,batch_size=128,depths = [128,64,1],x_size=(28,28)):
        self.reuse = False
        self.x_size = x_size
        self.batch_size = batch_size
        self.depths = depths

    def __call__(self,z,training=False,name =''):
        inputs = tf.convert_to_tensor(z)
        with tf.variable_scope('g',reuse = self.reuse):
            # reshape from inputs
            with tf.variable_scope('dense1'):
                outputs = tf.layers.dense(inputs,(int)(self.x_size[0]/4)*(int)(self.x_size[1]/4)*self.depths[0],activation = tf.nn.relu,trainable = training)
            with tf.variable_scope('reshape'):
                outputs = tf.reshape(outputs,[self.batch_size,(int)(self.x_size[0]/4),(int)(self.x_size[1]/4),self.depths[0]])
            with tf.variable_scope('deconv1'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv2'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2),activation = tf.nn.sigmoid, padding='SAME')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'g')
        return outputs


class Discriminator:
    def __init__(self,batch_size=128,depths = [64,128],x_size=(28,28)):
        self.reuse = False
        self.batch_size = batch_size
        self.depths= depths
        self.x_size = x_size

    def __call__(self,x,training=False,name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        inputs = tf.reshape(x,[self.batch_size,self.x_size[0],self.x_size[1],-1])
        with tf.name_scope('d'+name), tf.variable_scope('d',reuse = self.reuse):
            # reshape from inputs
            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(inputs, self.depths[0], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('classify'):
                reshape = tf.reshape(outputs, [self.batch_size, -1])
                outputs_logits = tf.layers.dense(reshape, 1, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'd')
        return outputs_logits


class WGANCONV:
    def __init__(self,batch_size=128,
                    g_depths = [128,64,1],
                    d_depths = [64,128],
                    x_size = (28,28),
                    z_dim = 100):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.x_size = x_size
        self.g = Generator(batch_size=self.batch_size,x_size = self.x_size,depths = g_depths)
        self.d = Discriminator(batch_size=self.batch_size,depths = d_depths)
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

    wgan = WGANCONV()
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
        wgan = WGANCONV()
        images = wgan.sample_images()

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,os.path.join('./model','model.ckpt'))

        generated = sess.run(images)
        with open('train.jpg','wb')as f:
            f.write(generated)

if __name__ == '__main__':
    mnisttrain()









    
