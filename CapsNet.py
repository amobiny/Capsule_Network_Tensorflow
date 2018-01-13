import tensorflow as tf
from config import *
from utils import routing, squash
import numpy as np


class CapsNet:
    def __init__(self):
        self.X = tf.placeholder(shape=[None, args.img_w, args.img_h, args.n_ch], dtype=tf.float32, name="X")
        self.Y = tf.placeholder(shape=[None, args.n_cls], dtype=tf.float32, name="Y")
        self.mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")

        self.build_network()
        self.loss()
        self.accuracy_calc()
        self.train_op()
        self.summary_()

    def build_network(self):
        with tf.variable_scope('Conv1_layer'):
            conv1 = tf.layers.conv2d(self.X, name="conv1", **conv1_params)
            # [batch_size, 20, 20, 256]

        with tf.variable_scope('PrimaryCaps_layer'):
            conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
            # [batch_size, 6, 6, 256]
            caps1_raw = tf.reshape(conv2, (args.batch_size, caps1_n_caps, caps1_n_dims, 1), name="caps1_raw")
            # [batch_size, 1152, 8, 1]
            caps1_output = squash(caps1_raw, name="caps1_output")
            # [batch_size, 1152, 8, 1]

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            caps2_input = tf.reshape(caps1_output, shape=(args.batch_size, caps1_n_caps, 1, caps1_n_dims, 1))
            # [batch_size, 1152, 1, 8, 1]
            b_IJ = tf.zeros([args.batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32, name="b_ij")
            # [batch_size, 1152, 10, 1, 1]
            self.caps2_output = routing(caps2_input, b_IJ, caps2_n_dims)
            # [batch_size, 10, 16, 1]

        # Decoder
        with tf.variable_scope('Masking'):
            epsilon = 1e-9
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2_output), axis=2, keep_dims=True) + epsilon)
            # [batch_size, 10, 1, 1]

            y_prob_argmax = tf.to_int32(tf.argmax(self.v_length, axis=1))
            # [batch_size, 1, 1]
            self.y_pred = tf.reshape(y_prob_argmax, shape=(args.batch_size,))
            # [batch_size] (predicted labels)
            y_pred_ohe = tf.one_hot(self.y_pred, depth=caps2_n_caps)
            # [batch_size, 10] (one-hot-encoded predicted labels)

            reconst_targets = tf.cond(self.mask_with_labels,  # condition
                                      lambda: self.Y,  # if True (Training)
                                      lambda: y_pred_ohe,  # if False (Test)
                                      name="reconstruction_targets")
            # [batch_size, 10]

            caps2_output_masked = tf.multiply(tf.squeeze(self.caps2_output), tf.expand_dims(reconst_targets, -1))
            # [batch_size, 10, 16]

            decoder_input = tf.reshape(caps2_output_masked, [args.batch_size, -1])
            # [batch_size, 160]

        with tf.variable_scope('Decoder'):
            fc1 = tf.layers.dense(decoder_input, n_hidden1, activation=tf.nn.relu, name="FC1")
            # [batch_size, 512]
            fc2 = tf.layers.dense(fc1, n_hidden2, activation=tf.nn.relu, name="FC2")
            # [batch_size, 1024]
            self.decoder_output = tf.layers.dense(fc2, n_output, activation=tf.nn.sigmoid, name="FC3")
            # [batch_size, 784]

    def loss(self):
        # 1. The margin loss

        # max(0, m_plus-||v_c||)^2
        present_error = tf.square(tf.maximum(0., args.m_plus - self.v_length))
        # [batch_size, 10, 1, 1]

        # max(0, ||v_c||-m_minus)^2
        absent_error = tf.square(tf.maximum(0., self.v_length - args.m_minus))
        # [batch_size, 10, 1, 1]

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        present_error = tf.reshape(present_error, shape=(args.batch_size, -1))
        absent_error = tf.reshape(absent_error, shape=(args.batch_size, -1))

        T_c = self.Y
        # [batch_size, 10]
        L_c = T_c * present_error + args.lambda_val * (1 - T_c) * absent_error
        # [batch_size, 10]
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1), name="margin_loss")

        # 2. The reconstruction loss
        orgin = tf.reshape(self.X, shape=(args.batch_size, -1))
        squared = tf.square(self.decoder_output - orgin)
        self.reconstruction_err = tf.reduce_mean(squared, name="reconstruction_loss")

        # 3. Total loss
        self.total_loss = tf.add(self.margin_loss, args.alpha * self.reconstruction_err, name="total_loss")

    def accuracy_calc(self):
        correct_prediction = tf.equal(tf.to_int32(tf.argmax(self.Y, axis=1)), self.y_pred)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train_op(self):
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.total_loss, name="training_op")

    def summary_(self):
        recon_img = tf.reshape(self.decoder_output, shape=(args.batch_size, args.img_w, args.img_h, args.n_ch))
        summary_list = [tf.summary.scalar('Loss/margin_loss', self.margin_loss),
                        tf.summary.scalar('Loss/reconstruction_loss', self.reconstruction_err),
                        tf.summary.image('original', self.X),
                        tf.summary.image('reconstructed', recon_img)]
        self.summary_now = tf.summary.merge(summary_list)
