import tensorflow as tf
import numpy as np
from PIL import Image

class init_Session():
    def __init__(self,sess):
        self.sess=sess
    def __enter__(self):
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.coord.request_stop()
        self.coord.join(self.threads)

def leakyrelu(x,leak=0.1):
    return tf.maximum(x, leak * x)  # 二者返回一个最大的

class WGAN_GP(object):

    def __init__(self,):
        self.LAMBDA=1
        self.batch_size=32
        self.min_width=6
        self.min_height = 6 #图片经过卷积核压缩的最小长度和宽度
        self.queue_size=800
        self.capacity=2*self.batch_size+self.queue_size
        self.learn_rate=0.001
        self.steps=3000000
        self.noise_dim=100
        self.imagepaths = tf.train.match_filenames_once("faces/*.jpg")
        self.filename_queue = tf.train.string_input_producer(self.imagepaths, shuffle=True)
        self.reader=tf.WholeFileReader()
        self.summarypath="summary"
        self.model_path="model/"
        self.output_impath="output/"
        self.create_impath='create_images/'

    def get_images(self):
        _,x = self.reader.read(self.filename_queue)
        image=tf.image.convert_image_dtype(
            tf.image.decode_jpeg(x, channels=3),dtype=tf.uint8)
        image = tf.cast(image, tf.float32)
        image=tf.reshape(image,[96,96,3])  #读入维度是?,?,3所以要改一下
        images=tf.train.shuffle_batch([image],
                               batch_size=self.batch_size,
                               capacity=self.capacity,
                               min_after_dequeue=320, name="real_images")
        return images

    def batch_histogram(self,varlist):
        for name,x in varlist:
            tf.summary.histogram(name,x)

    def generator(self,inputs,is_train,reuse):
        with tf.variable_scope('generator',reuse=reuse):
            fc=tf.layers.dense(
                inputs=inputs,
                units=self.min_height*self.min_width*512,
                activation=leakyrelu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name='fc',
            )

            feature_map=tf.reshape(fc,[self.batch_size,self.min_width,self.min_height,512])
            bn1= tf.layers.batch_normalization(
                inputs=feature_map,
                training=is_train,
                name='bn1'
            )
            deconv1=tf.layers.conv2d_transpose(
                inputs=bn1,
                filters=256,
                kernel_size=(3,3),
                strides=(2,2),
                padding='same',
                activation=leakyrelu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name='deconv1'
            )
            bn2 = tf.layers.batch_normalization(
                inputs=deconv1,
                training=is_train,
                name='bn2'
            )
            deconv2 = tf.layers.conv2d_transpose(
                inputs=bn2,
                filters=128,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                activation=leakyrelu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name = 'deconv2'
            )
            bn3= tf.layers.batch_normalization(
                inputs=deconv2,
                training=is_train,
                name='bn3'
            )
            deconv3=tf.layers.conv2d_transpose(
                inputs=bn3,
                filters=64,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                activation=leakyrelu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name='deconv3'
            )
            bn4 = tf.layers.batch_normalization(
                inputs=deconv3,
                training=is_train,
                name='bn4'
            )
            deconv4 = tf.layers.conv2d_transpose(
                inputs=bn4,
                filters=3,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                activation=tf.nn.tanh,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name='deconv4'
            )
            output=deconv4*127.5+127.5
        return output

    def discriminator(self,inputs,is_train,reuse):
        with tf.variable_scope('discriminator',reuse=reuse):
            conv1=tf.layers.conv2d(
                inputs=inputs,
                filters=64,
                kernel_size=(3,3),
                strides=(2,2),
                padding='same',
                activation=leakyrelu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name='conv1',
            )
            bn1 = tf.layers.batch_normalization(
                inputs=conv1,
                training=is_train,
                name='bn1',
            )
            conv2 = tf.layers.conv2d(
                inputs=bn1,
                filters=128,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                activation=leakyrelu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name='conv2',
            )
            bn2 = tf.layers.batch_normalization(
                inputs=conv2,
                training=is_train,
                name='bn2'
            )
            conv3= tf.layers.conv2d(
                inputs=bn2,
                filters=256,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                activation=leakyrelu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name='conv3',
            )
            bn3 = tf.layers.batch_normalization(
                inputs=conv3,
                training=is_train,
                name='bn3'
            )
            conv4= tf.layers.conv2d(
                inputs=bn3,
                filters=512,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                activation=leakyrelu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name='conv4',
            )
            bn4 = tf.layers.batch_normalization(
                inputs=conv4,
                training=is_train,
                name='bn4'
            )
            fc=tf.reshape(bn4,[self.batch_size,512*self.min_width*self.min_height])
            output=tf.layers.dense(
                inputs=fc,
                units=1,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name='output',
            )
        return output

    def saveImages(self,images,step,path):
        for i in range(self.batch_size):
            img= Image.fromarray(np.around(images[i]).astype('uint8'))
            img.save(path+str(step)+'  '+str(i)+'.jpg')

    def train(self):
        def optimizer(loss,var_list,n=1.0):
            step = tf.Variable(0, trainable=False)
            decay = 0.95
            num_decay_steps = 50
            learning_rate = tf.train.exponential_decay(
                self.learn_rate*n,
                step,
                num_decay_steps,
                decay,
                staircase=True
            )
            optimizer =tf.train.RMSPropOptimizer(learning_rate).minimize(
                loss,
                global_step=step,
                var_list=var_list)
            return optimizer

        images=self.get_images()
        alpha=tf.placeholder(tf.float32, [self.batch_size,1, 1, 1], name='alpha')
        is_train=tf.placeholder(tf.bool, name='is_train')

        noise = tf.placeholder( tf.float32, [self.batch_size,self.noise_dim], name='z')

        G =self.generator(noise,is_train,False)

        D_real= self.discriminator(images,is_train,reuse=False)
        D_fake= self.discriminator(G,is_train,reuse=True)

        loss_D = tf.subtract(tf.reduce_mean(D_real), tf.reduce_mean(D_fake), name='loss_D')
        loss_G = tf.reduce_mean(D_fake, name='loss_G')

        interpolates=alpha*(G-images)+images
        y_=self.discriminator(interpolates,is_train=is_train, reuse=True)
        gradients = tf.gradients(y_, [interpolates])[0]
        gradients=tf.reshape(gradients,[self.batch_size,96*96*3])
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        loss_D += self.LAMBDA* gradient_penalty

        vars = tf.trainable_variables()

        d_params = [v for v in vars if 'discriminator' in v.name]
        g_params = [v for v in vars if 'generator' in v.name]
        G_Op=optimizer(loss_G,g_params,1.5)
        D_Op=optimizer(loss_D,d_params)


        tf.summary.image("image", G)
        self.batch_histogram([("loss_D", loss_D), ("loss_G", loss_G)])
        merged = tf.summary.merge_all()


        with tf.Session() as sess:

            saver=tf.train.Saver()
            summary_writer = tf.summary.FileWriter(self.summarypath, sess.graph)
            with init_Session(sess):

                for i in range(self.steps):
                    datas = np.random.normal(1,100,[self.batch_size,self.noise_dim])
                    train_alpha=np.random.uniform(0.0,1.0,size=[self.batch_size,1,1,1]).astype(np.float32)
                    if i%16==0:

                        G_images,summary= sess.run([G,merged], {noise: datas,alpha:train_alpha,is_train:False})
                        summary_writer.add_summary(summary,i)
                        saver.save(sess, self.model_path + 'model.ckpt', global_step=i)
                        self.saveImages(G_images,i,self.output_impath)

                    __,ld,_,lg=sess.run(  [D_Op,loss_D,G_Op,loss_G]  ,
                                          {noise:datas,alpha:train_alpha,is_train:True}  )
                    print("\nloss_D", ld, "   loss_G", lg, i)
            summary_writer.close()

    def create(self):

        z= tf.placeholder(tf.float32, shape=[None,self.noise_dim],name='z')
        is_train = tf.placeholder(tf.bool, name='is_train')
        fake_image = self.generator(z, is_train, False)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            model_path = tf.train.latest_checkpoint(self.model_path,latest_filename=None)
            saver.restore(sess, model_path)
            test_noise = np.random.normal(1, 100, size=[self.batch_size,self.noise_dim]).astype(np.float32)
            [images] = sess.run([fake_image], feed_dict={z: test_noise, is_train:False})
            self.saveImages(images,-1,self.create_impath)

d=WGAN_GP()
d.train()
# d.create()