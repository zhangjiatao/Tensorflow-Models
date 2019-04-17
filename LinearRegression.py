import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

class LinearRegression(object):

    def __init__(self):

        # set params
        self._sess = tf.Session()
        self._learning_rate = 0.0001

        self._build_input()
        self._build_vars()

        # loss_op
        self.loss_op = self._build_loss()

        # init_op
        self.init_op = tf.global_variables_initializer()
        self._sess.run(self.init_op)

    def _build_input(self):
        """
            用来创建模型的输入，一般都为placeholder类型
        """
        self._X = tf.placeholder(tf.float32, name = "X")
        self._Y = tf.placeholder(tf.float32, name = "Y")

    def _build_vars(self):
        """
            用来创建模型的参数(学习参数)，一般都为Variable类型
        """
        self._W = tf.Variable(0.0, name = 'Weight')
        self._b = tf.Variable(0.0, name = 'bias')


    def _inference(self):
        """
            用于定义模型的计算过程，但从最后的角度理解，就是为了计算loss函数值
        """
        return self._W * self._X + self._b

    def _build_loss(self):
        """
            用于定义模型的loss值
        """
        Y_predict = self._inference()
        return tf.squared_difference(self._Y, Y_predict)

    def train(self, X, Y):
        """
            定义模型的训练操作，其实也就是整个代码的编写目标，优化loss值
        """
        train_op =  tf.train.GradientDescentOptimizer(self._learning_rate).minimize(self.loss_op)
        return self._sess.run([self.loss_op, train_op], feed_dict = {self._X : X, self._Y : Y})

    def predict(self):
        return self._sess.run([self._W, self._b])


if __name__ == "__main__":

    # set param
    tf.app.flags.DEFINE_string('log_dir', os.path.abspath('') + '/logs', 'Directory where log write to')
    tf.app.flags.DEFINE_integer('num_epochs', 20, 'the number of training epochs')
    FLAGS = tf.app.flags.FLAGS

    # create data (len = 15)
    x_list = []
    y_list = []
    for i in range(15):
        x_list.append(i)
    for x in x_list:
        y = 3 * x + 5
        if(y % 2 == 0):
            y = y + 2
        else:
            y = y - 3
        y_list.append(y)

    data = []
    for i in range(15):
        temp = []
        temp.append(x_list[i])
        temp.append(y_list[i])
        data.append(temp)

    data = np.array(data)


    with tf.Session() as sess:

        model = LinearRegression()
        writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)

        for epoch_num in range(FLAGS.num_epochs):

            for x, y in data:
                loss_value, _ = model.train(x, y)

            print('epoch : %d, loss : %f' % (epoch_num, loss_value))

            # save the values of weight and bias
            Weight, bias = model.predict()


    # Closing the writer.
    writer.close()
    sess.close()

    # plot
    Input_values = data[:, 0]
    Labels = data[:, 1]
    Prediction_values = data[:, 0] * Weight + bias
    plt.plot(Input_values, Labels, 'ro', label='main')
    plt.plot(Input_values, Prediction_values, label='Predicted')

    # Saving the result.
    plt.legend()
    plt.savefig('plot.png')
    plt.close()