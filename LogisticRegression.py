
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

class LogisticRegression(object):

    def __init__(self, num_features, num_classes):

        # set params
        self._sess = tf.Session()
        self._num_features = num_features
        self._num_classes = num_classes
        self._learning_rate = 0.0001

        # build input
        self._build_input()

        # loss op
        self.loss_op = self._build_loss()

        # train op
        self.train_op = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(self.loss_op)
        # global_step = tf.Variable(0, name="global_step", trainable=False)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        # gradients_and_variables = optimizer.compute_gradients(self.loss_op)
        # self.train_op = optimizer.apply_gradients(gradients_and_variables, global_step=global_step)


        # init op
        self.init_op = tf.global_variables_initializer()
        self._sess.run(self.init_op)


    def _build_input(self):
        self._image_place = tf.placeholder(tf.float32, shape = [None, self._num_features], name = 'image')
        self._label_place = tf.placeholder(tf.int32, shape = [None, ], name = 'gt')
        # features x depth if axis == -1
        # depth x features if axis == 0
        self._label_one_hot = tf.one_hot(self._label_place, depth = self._num_classes, axis= -1)
        self._dropout_param = tf.placeholder(tf.float32)

    def _inference(self):
        return tf.contrib.layers.fully_connected(inputs = self._image_place, num_outputs = self._num_classes, scope='fc')

    def _build_loss(self):
        logits = self._inference()
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= self._label_one_hot))

    def train(self, image, label):
        feed_dict = {self._image_place : image, self._label_place : label}
        return self._sess.run([self.loss_op, self.train_op], feed_dict= feed_dict)


if __name__ == "__main__":

    ##----------------------------------set Flags----------------------------------##
    tf.app.flags.DEFINE_string('log_dir', os.path.abspath('') + '/logs', 'Directory where log write to')
    tf.app.flags.DEFINE_integer('num_epochs', 20, 'the number of training epochs')
    tf.app.flags.DEFINE_integer('num_classes', 2, 'the number of classes')
    tf.app.flags.DEFINE_integer('num_features', 28 * 28, 'the fetures of the image')
    tf.app.flags.DEFINE_string('checkpoint_path', os.path.dirname(os.path.abspath(__file__)) + '/checkpoints', 'Directory where checkpoints are written to.')
    tf.app.flags.DEFINE_integer('max_num_checkpoint', 10, 'Maximum number of checkpoints that TensorFlow will keep.')
    tf.app.flags.DEFINE_integer('batch_size', 512, 'batch size')
    FLAGS = tf.app.flags.FLAGS

    ##----------------------------------data processing----------------------------------##
    mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=False)
    data = {}

    data['train/image'] = mnist.train.images
    data['train/label'] = mnist.train.labels
    data['test/image'] = mnist.test.images
    data['test/label'] = mnist.test.labels

    # Get only the samples with zero and one label for training.
    index_list_train = []
    for sample_index in range(data['train/label'].shape[0]):
        label = data['train/label'][sample_index]
        if label == 1 or label == 0:
            index_list_train.append(sample_index)

    # Reform the train data structure.
    data['train/image'] = mnist.train.images[index_list_train]
    data['train/label'] = mnist.train.labels[index_list_train]

    # Get only the samples with zero and one label for test set.
    index_list_test = []
    for sample_index in range(data['test/label'].shape[0]):
        label = data['test/label'][sample_index]
        if label == 1 or label == 0:
            index_list_test.append(sample_index)

    # Reform the test data structure.
    data['test/image'] = mnist.test.images[index_list_test]
    data['test/label'] = mnist.test.labels[index_list_test]

    # print shape
    print("[INFO] train/image shape", data['train/image'].shape)  # (11623, 784) 11623个训练样本，每个样本是28*28的图像
    print("[INFO] train/label shape", data['train/label'].shape)
    print("[INFO] test/image shape", data['test/image'].shape)  # (2115, 784)
    print("[INFO] test/label shape", data['test/label'].shape)


    ##----------------------------------loop----------------------------------##

    with tf.Session() as sess:

        model = LogisticRegression(FLAGS.num_features, FLAGS.num_classes)
        writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)

        for epoch in range(FLAGS.num_epochs):
            total_batch_num_train = int(data['train/image'].shape[0] / FLAGS.batch_size)

            for batch_num in range(total_batch_num_train):

                # create batch
                start_index = batch_num * FLAGS.batch_size
                end_index = (batch_num + 1) * FLAGS.batch_size
                image_batch, label_batch = data['train/image'][start_index:end_index], data['train/label'][start_index:end_index]

                # train
                loss_value, _ = model.train(image_batch, label_batch)

            print('[INFO] loss_value : %f' % loss_value)



