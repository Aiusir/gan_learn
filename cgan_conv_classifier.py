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
        x = tf.reshape(x,[self.batch_size,self.x_size[0],self.x_size[1],-1])
        with tf.name_scope('d'+name), tf.variable_scope('d',reuse = self.reuse):
            # reshape from inputs
            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(x, self.depths[0], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('dense'):
                reshape = tf.reshape(outputs, [self.batch_size, -1])
                outputs_logits = tf.layers.dense(reshape, 1, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'd')
        return outputs_logits

class CLASSIFIER:
    def __init__(self,batch_size=128,depths = [64,128],x_size=(28,28)):
        self.reuse = False
        self.batch_size = batch_size
        self.depths= depths
        self.x_size = x_size

    def __call__(self,x,training=False,name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        x = tf.reshape(x,[self.batch_size,self.x_size[0],self.x_size[1],-1])
        with tf.name_scope('c'+name), tf.variable_scope('c',reuse = self.reuse):
            # reshape from inputs
            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(x, self.depths[0], [5, 5], strides=(2, 2),activation = tf.nn.relu, padding='SAME')
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('dense1'):
                reshape = tf.reshape(outputs, [self.batch_size, -1])
                outputs = tf.layers.dense(reshape, 1024,activation = tf.nn.relu, name='outputs')
            with tf.variable_scope('dense2'):
                outputs = tf.layers.dense(outputs, 10, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'c')
        return outputs


class CGANCONVCLASSIFIER:
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
        self.c = CLASSIFIER(batch_size=self.batch_size,depths = d_depths)
    
    def loss(self,x_in,y_in):
        """build model,calculate losses.
        Args:
            x_in:4-D Tensor of shape '[batch,height,width,channels]'
            y_in:2-D Tensor of shape '[batch,label_num]'
        Returns:
            dict of each models loss.
        """
        generated = self.g(self.z,y_in,training = True)
        d_logits_real = self.d(x_in,training = True,name = 'd')
        d_logits_fake = self.d(generated,training = True,name = 'g')
        c_logits_real = self.c(x_in,training=True,name = 'cr')
        c_logits_fake = self.c(generated,training=True,name = 'cf')

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels = tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels = tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_fake+d_loss_real
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels = tf.ones_like(d_logits_fake)))
        c_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = c_logits_real,labels = y_in)) 
        c_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = c_logits_fake,labels = y_in))
        return {self.g:g_loss,self.d:d_loss,self.c:[c_loss_real,c_loss_fake]}

    def train(self,losses,learning_rate = 0.0002):
        """
        Args:
            losses dict
        Returns:
            train op.
        """
        d_opt_op = tf.train.AdamOptimizer(learning_rate).minimize(losses[self.d],var_list = self.d.variables)
        g_opt_op = tf.train.AdamOptimizer(learning_rate).minimize(losses[self.g],var_list = self.g.variables)
        c_opt_real_op = tf.train.AdamOptimizer(learning_rate).minimize(losses[self.c][0],var_list = self.c.variables)
        c_opt_fake_op = tf.train.AdamOptimizer(learning_rate).minimize(losses[self.c][1],var_list = self.g.variables)

        with tf.control_dependencies([g_opt_op,d_opt_op,c_opt_real_op,c_opt_fake_op]):
            return tf.no_op(name = 'train')

    def sample_images(self,inputs_y,x_size=(28,28),row=8,col =8,inputs = None):
        if inputs is None:
            inputs = self.z
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
    batch_size = 128
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

    cgan = CGANCONVCLASSIFIER()
    x_data = tf.placeholder(tf.float32,[batch_size,28*28],name = 'x_data')
    y_data = tf.placeholder(tf.float32,[batch_size,10],name = 'y_data')

    losses = cgan.loss(x_data,y_data)
    train_op = cgan.train(losses,learning_rate = 0.001)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for step in range(10000):
            x_in,y_in = mnist.train.next_batch(batch_size)
            _,g_loss_value,d_loss_value,c_loss_value = sess.run([train_op,losses[cgan.g],losses[cgan.d],losses[cgan.c]],feed_dict = {x_data:x_in,y_data:y_in})

            if step%1000==0:
                print step,g_loss_value,d_loss_value,c_loss_value

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

def mnisttest():
    batch_size = 128

    with tf.Session() as sess:
        cgan = CGANCONVCLASSIFIER()
        label = random.randint(0,9)
        y_input = tf.one_hot(label,depth = 10)
        y_titl = tf.tile(y_input,[batch_size])
        y_input = tf.reshape(y_titl,[batch_size,10])
        images = cgan.sample_images(y_input)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,os.path.join('./model','model.ckpt'))

        generated = sess.run(images)
        with open('train%d.jpg'%label,'wb')as f:
            f.write(generated)

if __name__ == '__main__':
    mnisttrain()









    
