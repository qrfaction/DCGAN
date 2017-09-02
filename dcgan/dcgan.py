import tensorflow as tf
import numpy as np
from PIL import Image

class ops(object):
    def conv2d(self,input, conv_depth,h=3, w=3,stddev=0.02,name="conv2d",striders=[1,2,2,1]):

        with tf.variable_scope(name):
            #卷积核权重
            filter_w = tf.get_variable('w', [w, h, input.get_shape()[-1], conv_depth],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            #卷积核扫描步长为2替代pool池进行降维处理
            conv = tf.nn.conv2d(input,filter_w, strides=striders, padding='SAME')

            biases = tf.get_variable('biases', [conv_depth], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

            return conv

    def deconv2d(self,input, output_shape,h=3, w=3,striders=[1,2,2,1], stddev=0.02,name="deconv2d", with_w=False):
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [w,h, output_shape[-1], input.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            deconv = tf.nn.conv2d_transpose(input, w, output_shape=output_shape,
                                        strides=striders,padding="SAME")

            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, biases)
            if with_w:
                return deconv, w, biases
            else:
                return deconv

    def leakyrelu(self,x, leak=0.2):
        return tf.maximum(x, leak * x)  # 二者返回一个最大的

    def fully_connect_layer(self,input,output_size, name="Linear", stddev=0.02, with_w=False):
        shape = input.get_shape().as_list()
        with tf.variable_scope(name):
            w = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.truncated_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias", [output_size],
                                   initializer=tf.constant_initializer(0.0))
            if with_w:
                return tf.matmul(input,w) + bias, w, bias
            else:
                return tf.matmul(input,w) + bias

    def batch_norm(self,x,is_training,reuse,name="batch_norm",epsilon=1e-6):
        with tf.variable_scope(name):
            return tf.contrib.layers.batch_norm(
                        x,
                        reuse=reuse,
                        updates_collections=None,
                        epsilon=epsilon,
                        scale=True,
                        is_training=is_training,
                        scope=name)

class initSess():
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

class DCGAN(object):

    def __init__(self,):
        self.ops=ops()
        self.batch_size=64
        self.min_width=6
        self.min_height = 6 #图片经过卷积核压缩的最小长度和宽度
        self.queue_size=800
        self.capacity=2*self.batch_size+self.queue_size
        self.learn_rate=0.01
        self.steps=3000000
        self.imagepaths = tf.train.match_filenames_once("images/*.jpg")
        self.filename_queue = tf.train.string_input_producer(self.imagepaths, shuffle=True)
        self.reader=tf.WholeFileReader()
        self.summarypath="summary"
        self.model_path="model_log/"
        self.create_impaths="create_images/"


    def get_image(self):
        _,x = self.reader.read(self.filename_queue)
        image=tf.image.convert_image_dtype(
            tf.image.decode_jpeg(x, channels=3),dtype=tf.uint8)
        image = tf.cast(image, tf.float32)
        image=tf.reshape(image,[96,96,3])  #读入维度是?,?,3所以要改一下
        return image

    def batch_histogram(self,varlist):
        for name,x in varlist:
            tf.summary.histogram(name,x)

    def generator(self,input,is_training,reuse):

        def getdecon(x,w_h,depth,name,nb_name,activation='relu'):
            o = self.ops.deconv2d(x, [self.batch_size, w_h[0],w_h[1], depth],name=name)
            o = self.ops.batch_norm(o, is_training=is_training, reuse=reuse, name=nb_name)
            if activation=="relu":
                return tf.nn.relu(o)
            else :
                return tf.nn.tanh(o)
        #生成各层  长  宽
        w_h=[(self.min_width*(2**i),self.min_height*(2**i)) for i in range(5)]

        with tf.variable_scope('generator',reuse=reuse):
            layer1=self.ops.fully_connect_layer(input,w_h[0][0]*w_h[0][1]*512,"g_l1")
            layer1=tf.reshape(layer1,[self.batch_size,w_h[0][0],w_h[0][0],512])
            layer1 = self.ops.batch_norm(layer1, is_training=is_training, reuse=reuse, name='g_bn0')
            layer1 = tf.nn.relu(layer1)

            decon1=getdecon(layer1,w_h[1],256,"g_dec1","g_dec1_nb")
            decon2 = getdecon(decon1, w_h[2],128, "g_dec2","g_dec2_nb")
            decon3 = getdecon(decon2, w_h[3], 64, "g_dec3","g_dec3_nb")
            decon4 = getdecon(decon3, w_h[4], 3, "g_dec4","g_dec4_nb","tanh")
        return decon4*127.5+127.5

    def discriminator(self,x,is_training,reuse):

        def get_conv(x,name,nb_name,depth):
            o = self.ops.conv2d(x, depth, name=name)
            o = self.ops.batch_norm(o, is_training, reuse, name=nb_name)
            o= self.ops.leakyrelu(o)
            return o
        with tf.variable_scope('decriminator',reuse=reuse):
            conv1=get_conv(x,"d_conv1","d_conv1_nb",64)
            conv2 = get_conv(conv1, "d_conv2", "d_conv2_nb",128)
            conv3 = get_conv(conv2, "d_conv3", "d_conv3_nb",256)
            conv4 = get_conv(conv3, "d_conv4", "d_conv4_nb",512)
            conv4=tf.reshape(conv4,[self.batch_size,512*self.min_width*self.min_height])
            fc=self.ops.fully_connect_layer(conv4,1,name="d_fc")
            t_f=tf.nn.sigmoid(fc)
        return t_f,fc

    def saveImage(self,name,image):
        img= Image.fromarray(np.around(image).astype('uint8'))
        img.save(self.create_impaths+name+'.jpg')

    def train(self):
        def optimizer(loss, var_list,name):
            step = tf.Variable(0, trainable=False)
            decay = 0.95
            num_decay_steps = 150
            learning_rate = tf.train.exponential_decay(
                self.learn_rate,
                step,
                num_decay_steps,
                decay,
                staircase=True
            )
            optimizer =tf.train.AdamOptimizer(learning_rate).minimize(
                loss,
                global_step=step,
                var_list=var_list,name=name)
            return optimizer
        image=self.get_image()
        images=tf.train.shuffle_batch([image],
                                      batch_size=self.batch_size,
                                      capacity=self.capacity,
                                      min_after_dequeue=320,name="real_images")

        with tf.variable_scope("G"):
            z = tf.placeholder(
                tf.float32, [self.batch_size,100], name='z')
            G =self.generator(z,True,False)

        with tf.variable_scope("D"):
            D1,fc1= self.discriminator(images,True,reuse=False)
            D2,fc2= self.discriminator(G,True, reuse=True)


        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('D/')]
        g_params = [v for v in vars if v.name.startswith('G/')]

        with tf.variable_scope("loss"):
            loss_D = tf.add(tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fc1, labels=tf.ones_like(fc1))),
                tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=fc2, labels=tf.zeros_like(fc2))), name="loss_D")
            loss_G = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fc2, labels=tf.ones_like(fc2)), name="loss_G")

            #参数分开训练
            G_Op=optimizer(loss_G,g_params,"G_Op")
            D_Op=optimizer(loss_D,d_params,"D_Op")


        tf.summary.image("image", G)
        self.batch_histogram([("loss_D", loss_D), ("loss_G", loss_G)])
        merged = tf.summary.merge_all()

        with tf.Session() as sess:

            # saver=tf.train.Saver()
            summary_writer = tf.summary.FileWriter(self.summarypath, sess.graph)
            with initSess(sess) as f:

                for i in range(self.steps):
                    datas = np.random.normal(1,100,[64,100])
                    # print(tf.all_variables())
                    if i%500==0:
                        #
                        G_images,summary= sess.run([G,merged], {z: datas})
                        summary_writer.add_summary(summary,i)
                        # saver.save(sess, self.model_path + 'model.ckpt', global_step=i)

                        for j in range(64):
                            self.saveImage(str(i)+'  ' +str(j), G_images[j])

                    ld,_,lg,__=sess.run([loss_D,D_Op,loss_G,G_Op],{z:datas})
                    lg,__=sess.run([loss_G,G_Op],{z:datas})
                    print("\nloss_D",ld,"   loss_G",lg,i)
            summary_writer.close()



a=DCGAN()
a.train()









