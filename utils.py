import os
from config import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

from download import download_fashion_mnist


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
    download_fashion_mnist(save_to=path)
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
    """
    Fetch the next batch of input images and labels
    :param x: all input images
    :param y: all labels
    :param start: first image number
    :param end: last image number
    :return: batch of images and their corresponding labels
    """
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def save_to():
    """
    Creating the handles for saving the results in a .csv file
    :return:
    """
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


def load_and_save_to(start_epoch, num_train_batch):
    """
    Loads the saved .csv files to continue training the model
    :return: the handles for saving into files and the minimum validation loss so far
    """
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
    acc_all = loss_all = pred_all = np.array([])
    num_batch = int(y.shape[0] / args.batch_size)
    for i in range(num_batch):
        start_val = i * args.batch_size
        end_val = start_val + args.batch_size
        x_b, y_b = get_next_batch(x, y, start_val, end_val)
        acc_batch, loss_batch, pred_batch = sess.run([model.accuracy, model.total_loss, model.y_pred],
                                                     feed_dict={model.X: x_b, model.Y: y_b})
        pred_all = np.append(pred_all, pred_batch)
        acc_all = np.append(acc_all, acc_batch)
        loss_all = np.append(loss_all, loss_batch)
    return np.mean(acc_all), np.mean(loss_all), pred_all


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
    fig.savefig(args.results + args.dataset + '/' + 'input_images.png')
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
    fig.savefig(args.results + args.dataset + '/' + 'reconstructed_images.png')
    plt.show()


def plot_adv_samples(x_orig, x_adv, y_true, y_pred_adv, y_pred, max_iter, epsilon, n_samples_per_class=3):
    idx = np.zeros((n_samples_per_class, args.n_cls)).astype(int)
    count = np.zeros(args.n_cls).astype(int)
    for i in range(y_pred_adv.shape[0]):
        # To plot only images classified correctly before, but are mistakenly classified
        # after the adversary attack
        if y_true[i] != y_pred_adv[i] and y_true[i] == y_pred[i] and count[y_true[i]] < n_samples_per_class:
            idx[count[y_true[i]], y_true[i]] = i
            count[y_true[i]] += 1
        else:
            continue
    idx = idx.reshape(-1, )
    fig = plt.figure(figsize=(10, n_samples_per_class * 1.2))
    for index in range(idx.size):
        plt.subplot(n_samples_per_class, args.n_cls, index + 1)
        plt.imshow(x_adv[idx[index]].reshape(args.img_w, args.img_h), cmap="gray")
        plt.title(str(y_pred_adv[idx[index]]))
        plt.xticks([])
        plt.yticks([])
    fig.savefig(args.results + args.dataset + '/' +
                'adv_attack_Xadv_iter_{0}_eps_{1}.png'.format(str(max_iter), str(epsilon)))
    plt.close(fig)
    fig = plt.figure(figsize=(10, n_samples_per_class * 1.2))
    for index in range(idx.size):
        plt.subplot(n_samples_per_class, args.n_cls, index + 1)
        plt.imshow(x_orig[idx[index]].reshape(args.img_w, args.img_h), cmap="gray")
        plt.xticks([])
        plt.yticks([])
    fig.savefig(args.results + args.dataset + '/' +
                'adv_attack_Xorig_iter_{0}_eps_{1}.png'.format(str(max_iter), str(epsilon)))
    plt.close(fig)
    fig = plt.figure(figsize=(10, n_samples_per_class * 1.2))
    for index in range(idx.size):
        plt.subplot(n_samples_per_class, args.n_cls, index + 1)
        plt.imshow((x_adv[idx[index]] - x_orig[idx[index]]).reshape(args.img_w, args.img_h), cmap="gray")
        plt.title(str(int(y_pred[idx[index]])) + '->' + str(y_pred_adv[idx[index]]))
        plt.xticks([])
        plt.yticks([])
    fig.savefig(args.results + args.dataset + '/' +
                'adv_attack_difference_iter_{0}_eps_{1}.png'.format(str(max_iter), str(epsilon)))
    plt.close(fig)


def plot_adv_curves(acc, loss, max_iter, epsilon):
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
    width, height = 10, 4
    fig.set_size_inches(width, height)

    ax = axs[0]
    ax.plot(epsilon, acc, '-o', color='k')
    ax.set_xlim([epsilon[0], epsilon[-1]])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Accuracy')
    ax.grid(color='lightgray', linestyle='-', linewidth=0.3)

    ax = axs[1]
    ax.plot(epsilon, loss, '-o', color='k')
    ax.set_xlim([epsilon[0], epsilon[-1]])
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Loss')
    ax.grid(color='lightgray', linestyle='-', linewidth=0.3)

    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)
    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=0.3, hspace=None)
    plt.show()
    fig.savefig(args.results + args.dataset + '/' +
                'adv_attack_curves_iter_{0}.png'.format(str(max_iter)))
