import pickle
import time
import sys
import numpy as np
import tensorflow as tf
from docopt import docopt

np.random.seed(3306)


# transforms sentence into a list of indices.
def get_idx_from_sent(sent, word_idx_map, maxlen, padding):
    x = []
    for i in range(padding):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < maxlen + 2 * padding:
        x.append(0)
    return x


# process datasets into [train, dev, test].
def make_idx_data(sentences, word_idx_map, maxlen, padding):
    train, dev, test = [], [], []
    for sen in sentences:
        s = get_idx_from_sent(sen['text'], word_idx_map, maxlen, padding)
        s.append(sen['y'])
        if sen['split'] == 'test':
            test.append(s)
        elif sen['split'] == 'train':
            train.append(s)
        elif sen['split'] == 'dev':
            dev.append(s)
    train = np.array(train, dtype='int')
    dev = np.array(dev, dtype='int')
    test = np.array(test, dtype='int')
    return [train, dev, test]


# train.
def train(datasets,
          W,
          maxlen,
          embedding_dims=300,
          dropout_input=0.9,
          dropout_hidden=0.8,
          batch_size=50,
          nb_epoch=15,
          nb_filter=100,
          filter_length=(3, 4, 5),
          hidden_dim=5,
          norm_lim=3,
          ):

    np.random.shuffle(datasets[0])
    np.random.shuffle(datasets[1])
    np.random.shuffle(datasets[2])

    X_train = np.asarray([d[:-1] for d in datasets[0]])
    Y_train = np.asarray([d[-1] for d in datasets[0]])
    X_dev = np.asarray([d[:-1] for d in datasets[1]])
    Y_dev = np.asarray([d[-1] for d in datasets[1]])
    X_test = np.asarray([d[:-1] for d in datasets[2]])
    Y_test = np.asarray([d[-1] for d in datasets[2]])

    def softmaxY(Y):
        newY = []
        for y in Y:
            tmpY = [0] * hidden_dim
            tmpY[y] = 1
            newY.append(tmpY)
        return np.asarray(newY)

    Y_train = softmaxY(Y_train)
    Y_dev = softmaxY(Y_dev)
    Y_test = softmaxY(Y_test)

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_dev shape:', X_dev.shape)
    print('Y_dev shape:', Y_dev.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)

    # initialize W in CNN.
    def conv_weight_variable(shape):
        initial = np.random.uniform(-0.01, 0.01, shape)
        conv_W = tf.Variable(initial, name='conv_W', dtype=tf.float32)
        return conv_W

    # initialize bias in CNN.
    def conv_bias_variable(shape):
        initial = tf.zeros(shape=shape)
        conv_b = tf.Variable(initial, name='conv_b', dtype=tf.float32)
        return conv_b

    # initialize W in fully connected layer.
    def fcl_weight_variable(shape):
        initial = tf.random_normal(shape=shape, stddev=0.01)
        fcl_W = tf.Variable(initial, name='fcl_W')
        fcl_W = fcl_W * tf.minimum(norm_lim / tf.norm(fcl_W), 1.0)
        return fcl_W

    # initialize bias in fully connected layer.
    def fcl_bias_variable(shape):
        initial = tf.zeros(shape=shape)
        fcl_b = tf.Variable(initial, name='fcl_b')
        return fcl_b

    # compute convolution.
    def conv1d(x, conv_W, conv_b):
        conv = tf.nn.conv1d(x,
                            conv_W,
                            stride=1,
                            padding='SAME',
                            name='conv')
        h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name='relu')
        return h

    # max-pooling.
    def max_pool(x):
        return tf.reduce_max(x, axis=1)

    def self_balanced_dropout(dropout_input, keep_prob):
        with tf.variable_scope(dropout_input.name.split(':')[0] + '/dropout'):
            hidden = dropout_input.get_shape().as_list()[-1]
            z_probability = tf.reduce_mean(tf.ones_like(dropout_input), axis=-1, keepdims=True) * keep_prob
            bias = tf.random_uniform(tf.shape(z_probability), 0, 1)
            z = tf.floor(z_probability + bias)
            dropout_token = tf.Variable(tf.truncated_normal([hidden], stddev=0.01), name='dropout_token')
            dropout_output = dropout_input * z + dropout_token * (1 - z)
        return tf.reshape(dropout_output, tf.shape(dropout_input))

    # set all states to default.
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    # input.
    x = tf.placeholder(tf.int32, [None, maxlen], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, hidden_dim], name='input_y')
    dropout_keep_prob_input = tf.placeholder(tf.float32, name='dropout_keep_prob_input')
    dropout_keep_prob_hidden = tf.placeholder(tf.float32, name='dropout_keep_prob_hidden')

    # embedding.
    with tf.device('/cpu:0'), tf.name_scope('embedding'):
        embedding_table = tf.Variable(W, name='embedding_table')
        embedded_words = tf.nn.embedding_lookup(embedding_table, x)

    embedded_words = self_balanced_dropout(embedded_words, dropout_keep_prob_input)

    # CNN.
    pooled_outputs = []
    for i in filter_length:
        with tf.name_scope('conv_maxpool_%s' % i):
            filter_shape = [i, embedding_dims, nb_filter]
            conv_W = conv_weight_variable(filter_shape)
            conv_b = conv_bias_variable([nb_filter])
            conv = conv1d(embedded_words, conv_W, conv_b)
            pooled = max_pool(conv)
            pooled_outputs.append(pooled)

    nb_filter_total = nb_filter * len(filter_length)
    h_pool = tf.concat(pooled_outputs, 1)
    h_pool_flat = tf.reshape(h_pool, [-1, nb_filter_total])

    # dropout.
    with tf.name_scope('dropout'):
        h_drop = self_balanced_dropout(tf.expand_dims(h_pool_flat, -1), dropout_keep_prob_hidden)
        h_drop = tf.squeeze(h_drop, axis=-1)

    # fully connected layer.
    with tf.name_scope('fcl'):
        fcl_W = fcl_weight_variable([nb_filter_total, hidden_dim])
        fcl_b = fcl_bias_variable([hidden_dim])
        fcl_output = tf.matmul(h_drop, fcl_W) + fcl_b
        y = tf.nn.softmax(fcl_output)
        y = tf.clip_by_value(y, clip_value_min=1e-6, clip_value_max=1 - 1e-6)

    # loss.
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y), reduction_indices=[1]))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-08)
    train_step = optimizer.minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # Train.
    tf.global_variables_initializer().run()

    best_accuracy = 0
    test_accuracy = 0
    for e in range(nb_epoch):
        epoch_starttime = time.time()
        train_loss = 0
        i = 0
        while i < len(X_train):
            if i + batch_size < len(X_train):
                batch_xs = X_train[i:i + batch_size]
                batch_ys = Y_train[i:i + batch_size]
            else:
                batch_xs = X_train[i:]
                batch_ys = Y_train[i:]
            i += batch_size
            r = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs,
                                                                 y_: batch_ys,
                                                                 dropout_keep_prob_input: dropout_input,
                                                                 dropout_keep_prob_hidden: dropout_hidden})
            train_loss += r[1] * len(batch_xs)
        train_loss = train_loss / len(X_train)

        def test(X, Y):
            i = 0
            correct = 0
            loss = 0
            while i < len(X):
                if i + batch_size < len(X):
                    batch_xs = X[i:i + batch_size]
                    batch_ys = Y[i:i + batch_size]
                else:
                    batch_xs = X[i:]
                    batch_ys = Y[i:]
                i += batch_size
                predictions, batch_loss = sess.run([correct_prediction, cross_entropy],
                                                   feed_dict={x: batch_xs,
                                                   y_: batch_ys,
                                                   dropout_keep_prob_input: 1.0,
                                                   dropout_keep_prob_hidden: 1.0})
                for p in predictions:
                    if p:
                        correct += 1
                loss += len(batch_xs) * batch_loss
            loss = loss / len(X)
            acc = correct / len(X)
            return acc, loss

        dev_accuracy, dev_loss = test(X_dev, Y_dev)

        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            test_accuracy, test_loss = test(X_test, Y_test)

        sys.stdout.write('Epoch: %d' % (e+1))
        sys.stdout.write('\tTrain Loss: %.6f' % train_loss)
        sys.stdout.write('\tDev Accuracy: %.6f' % dev_accuracy)
        sys.stdout.write('\tTest Accuracy: %.6f' % test_accuracy)
        sys.stdout.write('\tEpoch Time: %.1fs' % (time.time()-epoch_starttime))
        sys.stdout.write('\n')

    sess.close()

    return test_accuracy


# main function
def main():
    args = docopt('''
            Usage:
                cnn.py [options] <data_path> 

            Options:
                --padding NUM             pad a sentence with 0 in both sides [default: 4]
                --dropout_input NUM       keep probability of the input layer [default: 0.9]
                --dropout_hidden NUM      keep probability of the hidden layer [default: 0.7]
            ''')

    print('#########')
    print('Train CNN')
    print('#########')

    data_path = args['<data_path>']
    padding = int(args['--padding'])
    dropout_input = float(args['--dropout_input'])
    dropout_hidden = float(args['--dropout_hidden'])

    print('Loading Data...')
    data_file = open(data_path, 'rb')
    x = pickle.load(data_file)
    data_file.close()
    sentences, W, W2, word_idx_map, vocab, maxlen = x[0], x[1], x[2], x[3], x[4], x[5]
    print('Data Loaded!')

    final = []
    datasets = make_idx_data(sentences, word_idx_map, maxlen, padding)
    acc = train(datasets,
                W,
                dropout_input=dropout_input,
                dropout_hidden=dropout_hidden,
                maxlen=maxlen + 2 * padding)
    final.append(acc)
    print('Final Test Accuracy:' + str(np.mean(final)))


# entry point.
if __name__ == '__main__':
    main()
