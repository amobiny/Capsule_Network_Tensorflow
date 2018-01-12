from CapsNet import CapsNet
import tensorflow as tf
import numpy as np
from config import args
from utils import load_data, randomize, get_next_batch, save_to, load_and_save_to, evaluate
import os


def train(model):
    x_train, y_train, x_valid, y_valid = load_data(dataset='mnist', mode='train')
    num_train_batch = int(y_train.shape[0] / args.batch_size)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.checkpoint_path + args.dataset):
        os.makedirs(args.checkpoint_path + args.dataset)

    with tf.Session() as sess:
        if args.restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.checkpoint_path + args.dataset)
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_epoch = int(str(ckpt.model_checkpoint_path).split('-')[-1])
            fd_train, fd_val = load_and_save_to(start_epoch, num_train_batch)
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()
            fd_train, fd_val = save_to()
            start_epoch = 0
            best_loss_val = np.infty

        acc_batch_all = loss_batch_all = np.array([])
        train_writer = tf.summary.FileWriter(args.log_dir + args.dataset, sess.graph)
        for epoch in range(start_epoch, args.epoch):
            print('_____________________________________________________________________________')
            print('Training Epoch] #{}'.format(epoch + 1))
            x_train, y_train = randomize(x_train, y_train)
            for step in range(num_train_batch):
                start = step * args.batch_size
                end = (step + 1) * args.batch_size
                global_step = epoch * num_train_batch + step
                x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
                feed_dict_batch = {model.X: x_batch, model.Y: y_batch, model.mask_with_labels: True}
                if not (global_step % args.tr_disp_sum):
                    _, acc_batch, loss_batch, summary_tr = sess.run([model.train_op, model.accuracy,
                                                                     model.total_loss, model.summary_now],
                                                                    feed_dict=feed_dict_batch)
                    train_writer.add_summary(summary_tr, global_step)
                    acc_batch_all = np.append(acc_batch_all, acc_batch)
                    loss_batch_all = np.append(loss_batch_all, loss_batch)
                    mean_acc = np.mean(acc_batch_all)
                    mean_loss = np.mean(loss_batch_all)
                    summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Accuracy', simple_value=mean_acc)])
                    train_writer.add_summary(summary_tr, global_step)
                    summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Loss/total_loss', simple_value=mean_loss)])
                    train_writer.add_summary(summary_tr, global_step)

                    fd_train.write(str(global_step) + ',' + str(mean_acc) + ',' + str(mean_loss) + "\n")
                    fd_train.flush()
                    print("  Step #{0}, training loss: {1:.4f}, training accuracy: {2:.01%}".format(
                        global_step, mean_loss, mean_acc))
                    acc_batch_all = loss_batch_all = np.array([])
                else:
                    _, acc_batch, loss_batch = sess.run([model.train_op, model.accuracy, model.total_loss],
                                                        feed_dict=feed_dict_batch)
                    acc_batch_all = np.append(acc_batch_all, acc_batch)
                    loss_batch_all = np.append(loss_batch_all, loss_batch)

            # Run validation after each epoch
            acc_val, loss_val = evaluate(sess, model, x_valid, y_valid)
            fd_val.write(str(epoch+1) + ',' + str(acc_val) + ',' + str(loss_val) + '\n')
            fd_val.flush()
            print('-----------------------------------------------------------------------------')
            print("Epoch #{0}, Validation loss: {1:.4f}, Validation accuracy: {2:.01%}{3}".format(
                epoch + 1, loss_val, acc_val, "(improved)" if loss_val < best_loss_val else ""))

            # And save the model if it improved:
            if loss_val < best_loss_val:
                saver.save(sess, args.checkpoint_path + args.dataset + '/model.tfmodel', global_step=epoch + 1)
                best_loss_val = loss_val
        fd_train.close()
        fd_val.close()


def test(model):
    x_test, y_test = load_data(dataset='mnist', mode='test')
    fd_test = save_to()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_path + args.dataset)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        acc_test, loss_test = evaluate(sess, model, x_test, y_test)
        fd_test.write(str(acc_test) + ',' + str(loss_test) + '\n')
        fd_test.flush()
        print('-----------------------------------------------------------------------------')
        print("Test loss: {0:.4f}, Test accuracy: {1:.01%}".format(loss_test, acc_test))


def main(_):
    model = CapsNet()
    if args.mode == 'train':
        train(model)
    elif args.mode == 'test':
        test(model)


if __name__ == "__main__":
    tf.app.run()
