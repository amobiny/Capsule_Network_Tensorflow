from CapsNet import CapsNet
import tensorflow as tf
import numpy as np
from config import args
from utils import load_data, randomize, get_next_batch, save_to, load_and_save_to, evaluate, reconstruct_plot, \
    plot_adv_samples, plot_adv_curves
import os


def train(model):
    x_train, y_train, x_valid, y_valid = load_data(dataset=args.dataset, mode='train')
    print('Data set Loaded')
    num_train_batch = int(y_train.shape[0] / args.batch_size)
    if not os.path.exists(args.checkpoint_path + args.dataset):
        os.makedirs(args.checkpoint_path + args.dataset)

    with tf.Session() as sess:
        if args.restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.checkpoint_path + args.dataset)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model Restored')
            start_epoch = int(str(ckpt.model_checkpoint_path).split('-')[-1])
            fd_train, fd_val, best_loss_val = load_and_save_to(start_epoch, num_train_batch)
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()
            print('All variables initialized')
            fd_train, fd_val = save_to()
            start_epoch = 0
            best_loss_val = np.infty
        print('Start Training')
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
            acc_val, loss_val, _ = evaluate(sess, model, x_valid, y_valid)
            fd_val.write(str(epoch + 1) + ',' + str(acc_val) + ',' + str(loss_val) + '\n')
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
    x_test, y_test = load_data(dataset=args.dataset, mode='test')
    print('Data set Loaded')
    fd_test = save_to()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_path + args.dataset)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model Restored')
        acc_test, loss_test, _ = evaluate(sess, model, x_test, y_test)
        fd_test.write(str(acc_test) + ',' + str(loss_test) + '\n')
        fd_test.flush()
        print('-----------------------------------------------------------------------------')
        print("Test loss: {0:.4f}, Test accuracy: {1:.01%}".format(loss_test, acc_test))


def visualize(model, n_samples=5):
    x_test, y_test = load_data(dataset=args.dataset, mode='test')
    sample_images, sample_labels = x_test[:args.batch_size], y_test[:args.batch_size]
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_path + args.dataset)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        feed_dict_samples = {model.X: sample_images, model.Y: sample_labels}
        decoder_out, y_pred = sess.run([model.decoder_output, model.y_pred],
                                       feed_dict=feed_dict_samples)
    reconstruct_plot(sample_images, sample_labels, decoder_out, y_pred, n_samples)


def adv_attack(model, max_epsilon, max_iter):
    x_test, y_test = load_data(dataset=args.dataset, mode='test')
    print('Data set Loaded')
    all_acc = all_loss = np.array([])
    epsilon = tf.placeholder(shape=[], dtype=tf.float32, name="epsilon")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_path + args.dataset)

    # FGSM and Basic iteration (iterative version of FGSM; i.e. max_iter>1)
    dy_dx, = tf.gradients(model.total_loss, model.X)
    x_adv = tf.stop_gradient(model.X + epsilon * tf.sign(dy_dx))
    X_adv = tf.clip_by_value(x_adv, 0., 1.)

    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model Restored')
        num_batch = y_test.shape[0] / args.batch_size
        for eps in max_epsilon:     # loop over epsilon values
            iter_eps = eps
            x_adv_all = np.zeros((0, args.img_w, args.img_h, args.n_ch))
            eps /= max_iter
            for i in range(num_batch):      # loop over input batches
                start_val = i * args.batch_size
                end_val = start_val + args.batch_size
                x_adv_batch, y_batch = get_next_batch(x_test, y_test, start_val, end_val)
                for _ in range(max_iter):   # iterations
                    x_adv_batch = sess.run(X_adv, feed_dict={model.X: x_adv_batch, model.Y: y_batch, epsilon: eps})
                x_adv_all = np.concatenate((x_adv_all, x_adv_batch))
            acc_adv, loss_adv, y_pred_adv = evaluate(sess, model, x_adv_all, y_test)
            _, _, y_pred = evaluate(sess, model, x_test, y_test)
            print("Epsilon={0}, Test loss: {1:.4f}, Test accuracy: {2:.01%}".format(iter_eps, loss_adv, acc_adv))
            plot_adv_samples(x_test, x_adv_all,
                             np.argmax(y_test, axis=1), y_pred_adv.astype(int), y_pred,
                             max_iter, iter_eps, n_samples_per_class=5)
            all_acc = np.append(all_acc, acc_adv)
            all_loss = np.append(all_loss, loss_adv)
        plot_adv_curves(all_acc, all_loss, max_iter, max_epsilon)


def main(_):
    model = CapsNet()
    if args.mode == 'train':
        train(model)
    elif args.mode == 'test':
        test(model)
    elif args.mode == 'visualize':
        visualize(model, n_samples=args.n_samples)
    elif args.mode == 'adv_attack':
        adv_attack(model, max_epsilon=args.max_eps, max_iter=args.max_iter)


if __name__ == "__main__":
    tf.app.run()
