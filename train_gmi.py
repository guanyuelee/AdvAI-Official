# 设计一个GI神经模型，这个模型可以对任意一个分布进行训练，然后按照一定的epoch保存插值图片。
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow as tf
import numpy as np
import os
from lib import utils, layers
import matplotlib.pyplot as plt
import pickle as pkl

FLAGS = flags.FLAGS


class InterpolateOps:
    def __init__(self, x, x2, alpha_h, output, train_ops, ema):
        self.x = x
        self.x2 = x2
        self.alpha_h = alpha_h
        self.output = output
        self.train_ops = train_ops
        self.ema = ema


class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.np_vars = []
        self.ph_vars = []
        self.ops = []
        self.first_update = 0
        self.vars = None

    def apply(self, vars):
        count = 0
        self.vars = vars
        for var in vars:
            shape = var.get_shape().as_list()
            self.ph_vars.append(tf.placeholder(tf.float32, shape, 'ph_%d' % count))
            print('Layer %d: shape = ' % count, shape)
            self.ops.append(tf.assign(var, self.ph_vars[-1]))
            self.np_vars.append(np.zeros(shape, np.float))
            count += 1

    def update(self, sess):
        self.first_update += 1
        beta = self.beta
        if self.first_update == 1:
            beta = 0.0

        count = 0
        for var in self.vars:
            self.np_vars[count] = beta * self.np_vars[count] + (1 - beta) * sess.run(var)
            count += 1

        feed_dict = {}
        count = 0
        for ph in self.ph_vars:
            feed_dict[ph] = self.np_vars[count]
            count += 1

        sess.run(tuple(self.ops), feed_dict)


# Base class for managing the general interpolation
class BaseInterpolate:
    def __init__(self, prior, train_dir, **kwargs):
        self.prior = prior    # a latent dataset
        self.dims = prior.dims
        self.base_dir = os.path.join(train_dir, prior.name)
        self.train_dir = os.path.join(self.base_dir, self.experiment_name(**kwargs))
        self.params = kwargs
        self.sess = None
        self.cur_epochs = 0
        for dir in (self.base_dir, self.train_dir, self.image_dir, self.checkpoint_dir,
                    self.summary_dir):
            if not os.path.exists(dir):
                os.mkdir(dir)
                print('Create new folder: %s' % dir)

    def experiment_name(self, **kwargs):
        args = [x + str(y) for x, y in sorted(kwargs.items())]
        return '_'.join([self.__class__.__name__] + args)

    @property
    def image_dir(self):
        return os.path.join(self.train_dir, 'images')

    @property
    def checkpoint_dir(self):
        return os.path.join(self.train_dir, 'tf')

    @property
    def summary_dir(self):
        return os.path.join(self.checkpoint_dir, 'summaries')

    @property
    def tf_sess(self):
        return self.sess._tf_sess()

    def train_step(self, ops):
        x = self.prior.get_next(batch_size=FLAGS.batch_size) * self.n_scale
        self.sess.run(ops.train_ops, feed_dict={ops.x: x})
        if self.use_ema:
            ops.ema.update(self.tf_sess)

    def train(self, report_kepchs=64):
        print('report_kepochs = %d' % report_kepchs)
        batch_size = FLAGS.batch_size
        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()
            ops = self.model(**self.params)
            summary_hook = tf.train.SummarySaverHook(
                save_steps=(report_kepchs << 10)//batch_size,
                output_dir=self.summary_dir,
                summary_op=tf.summary.merge_all())
            report_hook = utils.HookReport(report_kepchs << 10, batch_size)
            stop_hook = tf.train.StopAtStepHook(last_step=1 + (FLAGS.total_kepochs << 10)//batch_size)

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=self.checkpoint_dir,
                    hooks=[stop_hook],
                    chief_only_hooks=[summary_hook, report_hook],
                    save_checkpoint_secs=600,
                    save_summaries_steps=0) as sess:
                self.sess = sess
                self.cur_epochs = self.tf_sess.run(global_step) * batch_size
                while not sess.should_stop():
                    self.train_step(ops)
                    self.cur_epochs = batch_size * self.tf_sess.run(global_step)

    def model(self, **kwargs):
        print('You should implement this method.')
        raise NotImplementedError

    # This method is for saving interpolation difference summary.
    def save_interpolation(self, ops, npts, npaths, n_ints=1000):
        # ops: the operator of the model
        # npts: the number of points to draw the distribution
        # npaths: the number of interpolation path.
        points = self.prior.get_points(npts=npts * 2)
        ppoints = self.prior.get_fix_points(npts=npaths * 2)
        batch_size = FLAGS.batch_size

        extra_pts = 100
        extra = self.prior.get_points(npts=extra_pts * 2)

        def batched_op(z1, z2, alpha):
            # get interpolated result
            return np.concatenate([self.tf_sess.run(ops.output,
                                                    {ops.x: z1[i:i + batch_size],
                                                     ops.x2: z2[i:i + batch_size],
                                                     ops.alpha_h: alpha[i:i + batch_size]})
                                   for i in range(0, z1.shape[0], batch_size)])

        # get distribution
        alpha = np.random.rand(npts, self.dims)
        int_points = batched_op(points[:npts], points[npts:], alpha)
        int_ppoints = []

        for i in range(n_ints):
            alpha = (i + 1) / (n_ints + 1)
            int_ppoints.append(batched_op(ppoints[:npaths],
                                          ppoints[npaths:],
                                          alpha * np.ones([npaths, self.dims])))

        int_ppoints = np.concatenate(int_ppoints, 0)

        extra_free = []
        extra_linear = []
        extra_n_ints = 20
        for i in range(extra_n_ints):
            alpha = (i + 1) / (extra_n_ints + 1)
            extra_free.append(batched_op(extra[:extra_pts],
                                         extra[extra_pts:],
                                         alpha * np.ones([extra_pts, self.dims])))
            extra_linear.append(extra[:extra_pts] * (1 - alpha) + extra[extra_pts:] * alpha)

        extra_linear = np.concatenate(extra_linear, axis=0)
        extra_free = np.concatenate(extra_free, axis=0)

        # save interpolation.
        save_num_points = 100
        save_interp_points = 10
        z1 = np.random.randn(save_num_points, 128)
        z2 = np.random.randn(save_num_points, 128)
        save_dict = {'save_interp_points': save_interp_points, 'z1': z1, 'z2': z2}
        interp_list = []
        for i in range(save_interp_points):
            alpha = i/(save_interp_points - 1)
            interp_list.append(batched_op(z1, z2, alpha * np.ones([save_num_points, 128])))

        save_dict['interp_list'] = interp_list

        with open(os.path.join(self.image_dir, str(self.cur_epochs >> 10) + '_float.pkl'), 'wb') as file_out:
            pkl.dump({'linear': extra_linear, 'free': extra_free}, file_out)

        with open(os.path.join(self.image_dir, str(self.cur_epochs >> 10) + '_interp_results.pkl'), 'wb') as file_out:
            pkl.dump(save_dict, file_out)

        if self.dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='.', label='prior')
            ax.scatter(int_ppoints[:, 0], int_ppoints[:, 1], int_ppoints[:, 2], c='b', marker='.', label='free')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.legend()
            plt.savefig(os.path.join(self.image_dir, str(self.cur_epochs >> 10) + '_path.png'))
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='.', label='prior')
            ax.scatter(int_points[:, 0], int_points[:, 1], int_points[:, 2], c='b', marker='.', label='free')
            plt.legend()
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.savefig(os.path.join(self.image_dir, str(self.cur_epochs >> 10) + '_distribution.png'))
            plt.close()

        elif self.dims > 3:
            fig = plt.figure()
            plt.plot(points[:, 0], points[:, 1], '.r', label='prior')
            plt.plot(int_ppoints[:, 0], int_ppoints[:, 1], ".b", label='free')
            plt.legend()
            plt.savefig(os.path.join(self.image_dir, str(self.cur_epochs >> 10) + '_path.png'))
            plt.close()

            fig = plt.figure()
            plt.plot(points[:, 0], points[:, 1], '.r', label='prior')
            plt.plot(int_points[:, 0], int_points[:, 1], ".b", label='free')
            plt.legend()
            plt.savefig(os.path.join(self.image_dir, str(self.cur_epochs >> 10) + '_distribution.png'))
            plt.close()

        d_np = plt.imread(os.path.join(self.image_dir, str(self.cur_epochs >> 10) + '_distribution.png'))
        p_np = plt.imread(os.path.join(self.image_dir, str(self.cur_epochs >> 10) + '_path.png'))
        return d_np, p_np


class GInterpolation(BaseInterpolate):
    # The definition of the GI interpolation model.
    def model(self, lr_int, lr_disc, int_hidden_units, int_hidden_layers, disc_hidden_layers, beta, n_scale,
              use_ema, dims, units, wa):
        x = tf.placeholder(tf.float32, [None, self.dims], 'x')
        x2 = tf.placeholder(tf.float32, [None, self.dims], 'x2')
        # interpolation coefficient
        alpha_h = tf.placeholder(tf.float32, [None, dims], 'alpha_h')
        self.n_scale = n_scale
        self.use_ema=True if use_ema==1 else False

        def interpolate(h, h_b, alpha):
            h = tf.reshape(h, [tf.shape(h)[0] * dims, 1])
            h_b = tf.reshape(h_b, [tf.shape(h_b)[0] * dims, 1])
            alpha_reshape = tf.reshape(alpha, [tf.shape(alpha)[0] * dims, 1])
            h_reshape = tf.concat((h, 1 - alpha_reshape), axis=1)
            h_reshape_b = tf.concat((h_b, alpha_reshape), axis=1)

            enc_layer_list = [units for i in range(int_hidden_layers)] + [int_hidden_units]
            h_encode = layers.fully_connected(h_reshape, 'int_enc', hidden_units=enc_layer_list)
            h_encode2 = layers.fully_connected(h_reshape_b, 'int_enc', hidden_units=enc_layer_list)
            h_mix = h_encode + h_encode2

            dec_layer_list = [units for i in range(int_hidden_layers)] + [1]
            h_bias = layers.fully_connected(h_mix, 'int_dec', hidden_units=dec_layer_list)
            res = (1 - alpha_reshape) * h + alpha_reshape * h_b + alpha_reshape * (1 - alpha_reshape) * h_bias

            res = tf.reshape(res, [-1, dims])
            return res

        def disc(z):
            z = tf.reshape(z, shape=[tf.shape(z)[0] * dims, 1])
            disc_layer_list = [units for i in range(disc_hidden_layers)] + [1]

            H_z = layers.fully_connected(z, 'disc_int', hidden_units=disc_layer_list)
            s_z = tf.reshape(H_z, [-1, dims])
            return s_z

        alpha = tf.random_uniform([tf.shape(x)[0], dims], 0, 1)
        alpha = 0.5 - tf.abs(alpha - 0.5)

        x_interpolate = interpolate(x * n_scale, x[::-1] * n_scale, alpha)/n_scale

        disc_loss = tf.reduce_mean(tf.abs(disc(x))) + \
                    tf.reduce_mean(tf.abs(disc(x_interpolate) - alpha))
        output = interpolate(x * n_scale, x2 * n_scale, alpha_h)/n_scale

        int_loss = tf.reduce_mean(tf.abs(disc(x_interpolate)))

        utils.HookReport.log_tensor(disc_loss, 'disc_loss')
        utils.HookReport.log_tensor(int_loss, 'int_loss')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        int_vars = tf.global_variables('int')
        disc_vars = tf.global_variables('disc')

        ema = EMA(beta=beta)
        ema.apply(int_vars)

        with tf.control_dependencies(update_ops):
            train_int = tf.train.AdamOptimizer(lr_int).minimize(
                wa * int_loss, var_list=int_vars)
            train_disc = tf.train.AdamOptimizer(lr_disc).minimize(
                disc_loss, var_list=disc_vars,
                global_step=tf.train.get_global_step())

        ops = InterpolateOps(x, x2, alpha_h,
                             output, tf.group(train_int, train_disc),
                             ema)

        def gen_images():
            print('ok gen_image')
            return self.save_interpolation(
                ops, npts=10000, npaths=15, n_ints=100)

        diff, line = tf.py_func(
            gen_images, [], [tf.float32] * 2)
        tf.summary.image('reconstruction', tf.expand_dims(diff, 0))
        tf.summary.image('interpolation', tf.expand_dims(line, 0))

        return ops


def main(argv):
    del argv
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Create folder %s' % FLAGS.train_dir)
    prior = utils.get_prior(FLAGS.prior, FLAGS.dims)
    model = GInterpolation(prior, FLAGS.train_dir,
                           lr_int=FLAGS.lr_int,
                           lr_disc=FLAGS.lr_disc,
                           int_hidden_units=FLAGS.int_hidden_units,
                           int_hidden_layers=FLAGS.int_hidden_layers,
                           disc_hidden_layers=FLAGS.disc_hidden_layers,
                           beta=FLAGS.beta,
                           n_scale=FLAGS.n_scale,
                           use_ema=FLAGS.use_ema,
                           dims = FLAGS.dims,
                           units=FLAGS.units,
                           wa=FLAGS.wa)
    model.train(report_kepchs=FLAGS.report_kepochs)


if __name__ == '__main__':
    flags.DEFINE_integer('total_kepochs', 1 << 14, 'Number of total k-epochs')
    flags.DEFINE_integer('report_kepochs', 1 << 6, 'Number of report k-epochs')
    flags.DEFINE_string('train_dir', './TRAIN_INT', 'Directory of the base folder')
    flags.DEFINE_string('prior', 'uniform', 'The prior distribution dataset')
    flags.DEFINE_float('beta', 0.9, 'beta')
    flags.DEFINE_integer('batch_size', 64, 'batch size')
    flags.DEFINE_float('lr_int', 1e-4, 'learning of interpolation module')
    flags.DEFINE_float('lr_disc', 1e-4, 'learning rate of disc module.')
    flags.DEFINE_integer('int_hidden_units', 100, 'number of hidden units in interpolate '
                                                   'module')
    flags.DEFINE_integer('int_hidden_layers', 2, 'number of hidden layers in interpolate'
                                                 'module')
    flags.DEFINE_integer('disc_hidden_layers', 2, 'number of hidden layers in interpolate'
                                                  'module')
    flags.DEFINE_float('n_scale', 2.0, 'scale')
    flags.DEFINE_integer('use_ema', 0, 'scale')
    flags.DEFINE_integer('dims', 2, 'dimension')
    flags.DEFINE_integer('units', 100, 'dimension')
    flags.DEFINE_float('wa', 0.05, 'wa')

    app.run(main)

