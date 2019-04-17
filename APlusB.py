"""
Basic Model
calculate a + b
"""

import tensorflow as tf
import os

class APB(object):

    def __init__(self):

        self._sess = tf.Session()

        self._build_constant()

        # op
        self.x = tf.add(self.a, self.b)
        self.y = tf.div(self.a, self.b)

    def _build_constant(self):
        self.a = tf.constant(10.3, name='a')
        self.b = tf.constant(12.4, name='b')


    def calculate(self):
        return self._sess.run([self.x, self.y])





if __name__ == "__main__":

    # set param
    tf.app.flags.DEFINE_string('log_dir', os.path.abspath('') + '/logs', 'Directory where log write to')

    FLAGS = tf.app.flags.FLAGS



    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)

        model = APB()
        print(model.calculate())

    # Closing the writer.
    writer.close()
    sess.close()