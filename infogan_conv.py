import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import os
import numpy as np

def sample_c(m,n,ind = -1):
    c=np.zeros([m,n])
    for i in range(m):
        if ind<0:
            ind = np.random.randint(10)
        c[i,ind] = 1
    return c

def sample_z(m,n):
    return np.random.uniform(-1.,1.,size=[m,n])



class Generator:
    def __init__(self,batch_size=128,depths = [128,64,1],x_size=(28,28)):
        self.reuse = False
        self.x_size = x_size
        self.batch_size = batch_size
        self.depths = depths

    def __call__(self,z,y,training=False,name =''):
        inputs = tf.concat([z,y],1)
        inputs = tf.convert_to_tensor(inputs)
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
                outputs_logits = tf.layers.dense(reshape, 1,name='outputs1')
                outputs = tf.layers.dense(reshape, 128, name='outputs2')
                outputs =  leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs3')
                outputs_prob = tf.layers.dense(outputs, 10, name='outputs4')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'd')
        return outputs_logits,outputs_prob


class INFOGANCONV:
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
    
    def loss(self,x_in,y_in,z_in):
        """build model,calculate losses.
        Args:
            x_in:4-D Tensor of shape '[batch,height,width,channels]'
            y_in:2-D Tensor of shape '[batch,label_num]'
        Returns:
            dict of each models loss.
        """
        generated = self.g(z_in,y_in,training = True)
        d_logits_real,_ = self.d(x_in,training = True,name = 'd')
        d_logits_fake,d_labels_fake = self.d(generated,training = True,name = 'g')

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels = tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels = tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_fake+d_loss_real
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels = tf.ones_like(d_logits_fake)))
        q_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_labels_fake,labels = y_in))
        return {self.g:g_loss,self.d:d_loss,"q_loss":q_loss}

    def train(self,losses,learning_rate = 0.0002):
        """
        Args:
            losses dict
        Returns:
            train op.
        """
        d_opt_op = tf.train.AdamOptimizer(learning_rate).minimize(losses[self.d],var_list = self.d.variables)
        g_opt_op = tf.train.AdamOptimizer(learning_rate).minimize(losses[self.g],var_list = self.g.variables)
        q_opt_op = tf.train.AdamOptimizer(learning_rate).minimize(losses["q_loss"],var_list = self.d.variables+self.g.variables)

        with tf.control_dependencies([g_opt_op,d_opt_op,q_opt_op]):
            return tf.no_op(name = 'train')

    def sample_images(self,inputs,inputs_y,x_size=(28,28),row=8,col =8):
        imgs = self.g(inputs,inputs_y,training = True)
        imgs = tf.reshape(imgs,[self.batch_size,x_size[0],x_size[1],1])
        imgs = tf.image.convert_image_dtype(imgs,tf.uint8)
        imgs = [img for img in tf.split(imgs,self.batch_size,axis = 0)]

        rows = []
        for i in range(row):
            rows.append(tf.concat(imgs[col*i+0:col*i+col],2))
        img = tf.concat(rows,1)
        return tf.image.encode_jpeg(tf.squeeze(img,[0]))

def mnisttrain():
    batch_size = 64
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

    infogan = INFOGANCONV(batch_size=batch_size)
    x_data = tf.placeholder(tf.float32,[batch_size,28*28],name = 'x_data')
    y_data = tf.placeholder(tf.float32,[batch_size,10],name = 'y_data')
    z_data = tf.placeholder(tf.float32,[batch_size,100],name = 'z_data')

    losses = infogan.loss(x_data,y_data,z_data)
    train_op = infogan.train(losses,learning_rate = 0.0001)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for step in range(10000):
            x_in,_ = mnist.train.next_batch(batch_size)
            y_in = sample_c(batch_size,10)
            z_in = sample_z(batch_size,100)
            _,g_loss_value,d_loss_value,q_loss_value = sess.run([train_op,losses[infogan.g],losses[infogan.d],losses['q_loss']],feed_dict = {x_data:x_in,y_data:y_in,z_data:z_in})

            if step%1000==0:
                print step,g_loss_value,d_loss_value,q_loss_value

            if step%10000==0:
                label = random.randint(0,9)
                y_input = tf.one_hot(label,depth = 10)
                y_titl = tf.tile(y_input,[batch_size])
                y_input = tf.reshape(y_titl,[batch_size,10])
                images = infogan.sample_images(z_in,y_input)
                generated = sess.run(images)
                with open('train%d_%d.jpg'%(step,label),'wb')as f:
                    f.write(generated)
        saver.save(sess,os.path.join('./model','model.ckpt'))

def mnisttest():
    batch_size = 128

    with tf.Session() as sess:
        infogan = INFOGANCONV()
        z_in = sample_z(batch_size,100)
        label = random.randint(0,9)
        y_input = tf.one_hot(label,depth = 10)
        y_titl = tf.tile(y_input,[batch_size])
        y_input = tf.reshape(y_titl,[batch_size,10])
        images = infogan.sample_images(z_in,y_input)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,os.path.join('./model','model.ckpt'))

        generated = sess.run(images)
        with open('train%d.jpg'%label,'wb')as f:
            f.write(generated)

if __name__ == '__main__':
    mnisttrain()









    
