import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

###################
# *** Global Variables
LOG_DIR = 'output'


def cnn_classifier(x, reuse):
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.name_scope('conv1'):
            conv1 = tf.layers.conv2d(
                x, filters=32, kernel_size=(5, 5),
                padding='same', activation=tf.nn.relu)
        with tf.name_scope('pool1'):
            pool1 = tf.layers.max_pooling2d(
                conv1, pool_size=(5, 5), strides=(3, 3))
        with tf.name_scope('fc2'):
            pool1_flat = tf.contrib.layers.flatten(pool1)
            fc2 = tf.layers.dense(pool1_flat, 1024)
        with tf.name_scope('output'):
            out = tf.layers.dense(fc2, 10)
    return out


def train():
    # Import data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print(type(mnist))

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None, 10], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    y = cnn_classifier(image_shaped_input, reuse=False)

    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = tf.losses.softmax_cross_entropy(y_, y)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(0.001)
        train_step = optimizer.minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correcr'):
            correct = tf.equal(tf.argmax(y, 1), y_)
        with tf.name_scope('accuracy'):
            accuracy = accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    sess = tf.Session()

    # Marged all summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test')

    # initializer
    init = tf.global_variables_initializer()
    sess.run(init)

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = mnist.train.next_batch(100)
        else:
            xs, ys = mnist.test.images, mnist.test.labels
        return {x: xs, y_: ys}
    for i in range(30):
        if i % 5 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run(
                [merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            summary, _ = sess.run([merged, train_step],
                                  feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


def main(argv):
    train()


if __name__ == '__main__':
    tf.app.run()
