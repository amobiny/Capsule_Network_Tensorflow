import time

from CapsNet import CapsNet
import tensorflow as tf
import numpy as np
from config import args
from utils import load_data, randomize, get_next_batch
import os


def train(model):
    x_train, y_train, x_valid, y_valid = load_data(dataset='mnist', mode='train')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with tf.Session() as sess:
        if args.restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            last_episode = int(str(ckpt.model_checkpoint_path).split('-')[-1])
            # all_acc_train, all_loss_train = load_results(args, last_episode, mode='train')
            # all_acc_test, all_loss_test = load_results(args, last_episode, mode='test')
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()

        acc_batch_all = loss_batch_all = np.array([])
        sum_count = 0
        train_writer = tf.summary.FileWriter(args.log_dir + '/train/', sess.graph)
        valid_writer = tf.summary.FileWriter(args.log_dir + '/valid/')
        for epoch in range(args.epoch):
            epoch_start_time = time.time()
            print('-----------------------------------------------------------------------------')
            print('Epoch: {}'.format(epoch + 1))
            x_train, y_train = randomize(x_train, y_train)
            step_count = int(len(x_train) / args.batch_size)
            for step in range(step_count):
                start = step * args.batch_size
                end = (step + 1) * args.batch_size
                x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
                feed_dict_batch = {model.X: x_batch, model.Y: y_batch, model.mask_with_labels: True}
                _, acc_batch, loss_batch = sess.run([model.train_op, model.accuracy, model.total_loss],
                                                    feed_dict=feed_dict_batch)
                acc_batch_all = np.append(acc_batch_all, acc_batch)
                loss_batch_all = np.append(loss_batch_all, loss_batch)

                if step > 0 and not (step % args.tr_disp_sum):
                    mean_acc = np.mean(acc_batch_all)
                    mean_loss = np.mean(loss_batch_all)
                    print(
                        "Step {0}, training loss: {1:.5f}, training accuracy: {2:.01f}%".format(step, mean_loss,
                                                                                                mean_acc))
                    summary_tr = sess.run(model.summary_(mean_acc, mean_loss), feed_dict=feed_dict_batch)
                    train_writer.add_summary(summary_tr, sum_count * args.tr_disp_sum)
                    sum_count += 1
                    acc_batch_all = loss_batch_all = np.array([])


def main(_):
    model = CapsNet()
    if args.mode:
        train(model)
    else:
        pass
        # evaluation(model)


if __name__ == "__main__":
    tf.app.run()
