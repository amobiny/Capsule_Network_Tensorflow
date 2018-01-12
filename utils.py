import h5py
import os
from config import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(mode='train'):
    """
    load the MNIST data
    :param mode: train or test
    :return: train and validation images and labels in train mode, test images and labels in test mode
            x: [#images, width, height, n_channels]
            y: [#images, #classes=10] (one_hot_encoded)
    """
    mnist = input_data.read_data_sets("data/mnist", one_hot=True)
    if mode == 'train':
        x_train, y_train, x_valid, y_valid = mnist.train.images, mnist.train.labels, \
                                             mnist.validation.images, mnist.validation.labels
        x_train = x_train.reshape((-1, args.img_w, args.img_h, args.n_ch)).astype(np.float32)
        x_valid = x_valid.reshape((-1, args.img_w, args.img_h, args.n_ch)).astype(np.float32)
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        x_test, y_test = mnist.test.images, mnist.test.labels
        x_test = x_test.reshape((-1, args.img_w, args.img_h, args.n_ch)).astype(np.float32)
        return x_test, y_test


def load_fashion_mnist(mode='train'):
    path = os.path.join('data', 'fashion-mnist')
    if mode == 'train':
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        x = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        y = loaded[8:].reshape(60000).astype(np.int32)

        x_train = x[:55000] / 255.
        y_train = y[:55000]
        y_train = (np.arange(args.n_cls) == y_train[:, None]).astype(np.float32)

        x_valid = x[55000:, ] / 255.
        y_valid = y[55000:]
        y_valid = (np.arange(args.n_cls) == y_valid[:, None]).astype(np.float32)
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        x_test = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        y_test = loaded[8:].reshape(10000).astype(np.int32)
        y_test = (np.arange(args.n_cls) == y_test[:, None]).astype(np.float32)
        return x_test / 255., y_test


def load_data(dataset, mode='train'):
    if dataset == 'mnist':
        return load_mnist(mode)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(mode)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def squash(s, epsilon=1e-7, name=None):
    """
    Squashing function corresponding to Eq. 1
    :param s: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    :param epsilon: To compute norm safely
    :param name:
    :return: A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    """
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=-1, keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


def routing(inputs, b_ij, out_caps_dim):
    """
    The routing algorithm
    :param inputs: A tensor with [batch_size, num_caps_in=1152, 1, in_caps_dim=8, 1] shape.
                  num_caps_in: the number of capsule in layer l (i.e. PrimaryCaps).
                  in_caps_dim: dimension of the output vectors of layer l (i.e. PrimaryCaps)
    :param b_ij: [batch_size, num_caps_in=1152, num_caps_out=10, 1, 1]
                num_caps_out: the number of capsule in layer l+1 (i.e. DigitCaps).
    :param out_caps_dim: dimension of the output vectors of layer l+1 (i.e. DigitCaps)

    :return: A Tensor of shape [batch_size, num_caps_out=10, out_caps_dim=16, 1]
            representing the vector output `v_j` in layer l+1.
    """
    # W: [num_caps_in, num_caps_out, len_u_i, len_v_j]
    W = tf.get_variable('W', shape=(1, inputs.shape[1].value, b_ij.shape[2].value, inputs.shape[3].value, out_caps_dim),
                        dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=args.stddev))

    inputs = tf.tile(inputs, [1, 1, b_ij.shape[2].value, 1, 1])
    # input => [batch_size, 1152, 10, 8, 1]

    W = tf.tile(W, [args.batch_size, 1, 1, 1, 1])
    # W => [batch_size, 1152, 10, 8, 16]

    u_hat = tf.matmul(W, inputs, transpose_a=True)
    # [batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # For r iterations do
    for r_iter in range(args.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            c_ij = tf.nn.softmax(b_ij, dim=2)
            # [batch_size, 1152, 10, 1, 1]

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == args.iter_routing - 1:
                s_j = tf.multiply(c_ij, u_hat)
                # [batch_size, 1152, 10, 16, 1]
                # then sum in the second dim
                s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)
                # [batch_size, 1, 10, 16, 1]
                v_j = squash(s_j)
                # [batch_size, 1, 10, 16, 1]

            elif r_iter < args.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_j = tf.multiply(c_ij, u_hat_stopped)
                s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)
                v_j = squash(s_j)
                v_j_tiled = tf.tile(v_j, [1, inputs.shape[1].value, 1, 1, 1])
                # [batch_size, 1152, 10, 16, 1]

                # then matmul in the last two dim: [16, 1].T x [16, 1] => [1, 1]
                u_produce_v = tf.matmul(u_hat_stopped, v_j_tiled, transpose_a=True)
                # [batch_size, 1152, 10, 1, 1]

                b_ij += u_produce_v
    return tf.squeeze(v_j, axis=1)
    # [batch_size, 10, 16, 1]


def save_to():
    if not os.path.exists(args.results):
        os.mkdir(args.results)
    if not os.path.exists(args.results + args.dataset):
        os.mkdir(args.results + args.dataset)
    if args.mode == 'train':
        train_path = args.results + args.dataset + '/' + 'train.csv'
        val_path = args.results + args.dataset + '/' + 'validation.csv'

        if os.path.exists(train_path):
            os.remove(train_path)
        if os.path.exists(val_path):
            os.remove(val_path)

        f_train = open(train_path, 'w')
        f_train.write('step,accuracy,loss\n')
        f_val = open(val_path, 'w')
        f_val.write('epoch,accuracy,loss\n')
        return f_train, f_val
    else:
        test_path = args.results + args.dataset + '/test.csv'
        if os.path.exists(test_path):
            os.remove(test_path)
        f_test = open(test_path, 'w')
        f_test.write('accuracy,loss\n')
        return f_test


def load_and_save_to(epoch, num_train_batch):

    train_path = args.results + args.dataset + '/' + 'train.csv'
    val_path = args.results + args.dataset + '/' + 'validation.csv'
    # finding the minimum validation loss so far
    f_ = open(val_path, 'r')
    lines = f_.readlines()
    a = np.genfromtxt(lines[-1:], delimiter=',')
    min_loss = np.min(a[1:, 2])
    # loading the .csv file to continue recording the values
    f_train = open(train_path, 'a')
    f_val = open(val_path, 'a')
    return f_train, f_val, min_loss


def evaluate(sess, model, x, y):
    acc_all = loss_all = np.array([])
    num_batch = y.shape[0] / args.batch_size
    for i in range(num_batch):
        start_val = i * args.batch_size
        end_val = start_val + args.batch_size
        x_b, y_b = get_next_batch(x, y, start_val, end_val)
        acc_batch, loss_batch = sess.run([model.accuracy, model.total_loss],
                                         feed_dict={model.X: x_b, model.Y: y_b})
        acc_all = np.append(acc_all, acc_batch)
        loss_all = np.append(loss_all, loss_batch)
    return np.mean(acc_all), np.mean(loss_all)


def reconstruct_plot(x, y, x_reconst, y_pred, n_samples):
    fashion_mnist_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    sample_images = x.reshape(-1, args.img_w, args.img_h)
    reconst = x_reconst.reshape([-1, args.img_w, args.img_h])

    fig = plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(1, n_samples, index + 1)
        plt.imshow(sample_images[index], cmap="binary")
        if args.dataset == 'mnist':
            plt.title("Label:" + str(np.argmax(y[index])))
        elif args.dataset == 'fashion-mnist':
            plt.title("Label:" + fashion_mnist_labels[np.argmax(y[index])])
        plt.axis("off")
    fig.savefig(args.results + args.dataset + '/' + 'input_images')
    plt.show()

    fig = plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(1, n_samples, index + 1)
        plt.imshow(reconst[index], cmap="binary")
        if args.dataset == 'mnist':
            plt.title("Predicted:" + str(y_pred[index]))
        elif args.dataset == 'fashion-mnist':
            plt.title("Pred:" + fashion_mnist_labels[y_pred[index]])
        plt.axis("off")
    fig.savefig(args.results + args.dataset + '/' + 'reconstructed_images')
    plt.show()
