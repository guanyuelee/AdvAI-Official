'''
GI + DRAE = GIDRAE. 1 dimensional.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from absl import app
from absl import flags
from lib import data, layers, train, utils, classifiers, eval
import numpy as np
from lib.mmd import mmd2
import os
from PIL import Image

from lib.inception_score import get_inception_score

FLAGS = flags.FLAGS


class AEOps(object):

    def __init__(self, x, h, h_b, label, encode, decode, ae, train_op, train_op_extra,
                 alpha_h, my_encode_mix, train_int_op, ema_op, classify_latent=None):
        self.x = x
        self.h = h
        self.h_b = h_b
        self.label = label
        self.encode = encode
        self.decode = decode
        self.ae = ae
        self.train_op = train_op
        self.train_op_extra = train_op_extra
        self.classify_latent = classify_latent
        self.alpha_h = alpha_h
        self.my_encode_mix = my_encode_mix
        self.train_int_op = train_int_op
        self.ema_op = ema_op


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

        var_values = sess.run(self.vars)
        count = 0
        for var in var_values:
            self.np_vars[count] = beta * self.np_vars[count] + (1 - beta) * var
            count += 1

        feed_dict = {}
        count = 0
        for ph in self.ph_vars:
            feed_dict[ph] = self.np_vars[count]
            count += 1

        sess.run(tuple(self.ops), feed_dict)


class GIDRAE2(train.AE):

    # Function to train the autoencoder and the discriminator.
    def train_int_step(self, data, ops):
        x = self.tf_sess.run(data)
        x, label = x['x'], x['label']

        shape = ops.h.get_shape().as_list()
        random_latents = np.random.standard_normal(size=(FLAGS.batch, shape[1], shape[2], shape[3]))

        self.sess.run(ops.train_int_op, feed_dict={ops.x: x, ops.label: label,
                                                   ops.h: self.n_scale * random_latents,
                                                   ops.alpha_h: self.get_alpha(batch=FLAGS.batch)})
        if self.use_ema:
            ops.ema_op.update(sess=self.tf_sess)

    def get_alpha(self, batch):
        dim = 1

        alpha = 0.5 * np.random.rand(batch, dim)

        return np.reshape(alpha, [batch, 1, 1, 1])

    def train_step(self, data, ops):
        x = self.tf_sess.run(data)
        x, label = x['x'], x['label']

        shape = ops.h.get_shape().as_list()
        random_latents = np.random.standard_normal(size=(FLAGS.batch, shape[1], shape[2], shape[3]))

        self.sess.run(ops.train_op, feed_dict={ops.x: x, ops.label: label,
                                               ops.h: random_latents,
                                               ops.alpha_h: self.get_alpha(batch=FLAGS.batch)})

    def model(self, latent, depth, scales, advweight, advdepth, reg, int_hidden_layers,
              int_hidden_units, disc_hidden_layers, beta, wgt_mmd, use_ema, n_scale,
              wgt_noise, wgt_fake):

        """
            :param latent: The number of channels in latent space.
            :param depth: The number of channels in the first conv operation.
            :param scales: The number of scales.
            :param advweight: The weight for disc.
            :param advdepth: The number of channels in the first conv operation in disc.
            :param reg: The ratio for combine reconstruction and the original images.
            :param int_hidden_layers: The number of layers in the interpolation module.
            :param int_hidden_units: The number of units in the hidden layer of interpolation module.
            :param disc_hidden_layers: The number of layers in the int_disc module.
            :param beta: The momentum of ema.
            :param wgt_mmd: weight for mmd regularization.
            :param use_ema: option to toggle using ema.
            :param n_scale: the scale of training interpolation.
            :param wgt_noise: weight for noise interpolation.
            :param wgt_fake: weight for fake interpolation.
            :return: The operation for manipulating the model.
        """

        dim = latent * FLAGS.latent_width ** 2
        x = tf.placeholder(tf.float32, [None, self.height, self.width, self.colors], 'x')
        l = tf.placeholder(tf.float32, [None, self.nclass], 'label')
        h = tf.placeholder(
            tf.float32,
            [None, self.height >> scales, self.width >> scales, latent], 'h')
        h_b = tf.placeholder(
            tf.float32,
            [None, self.height >> scales, self.width >> scales, latent], 'h_b')

        self.use_ema = True if use_ema == 1 else False
        self.n_scale = n_scale

        alpha_h = tf.placeholder(tf.float32, [None, 1, 1, 1], 'alpha_h')

        def encoder(x):
            return layers.encoder(x, scales, depth, latent, 'ae_enc')

        def decoder(h):
            v = layers.decoder(h, scales, depth, self.colors, 'ae_dec')
            return v

        def interpolate(h_a, h_b, alpha):
            alpha_dim = 1

            alpha_reshape = tf.reshape(alpha, shape=[tf.shape(alpha)[0], alpha_dim])

            h_reshape_a = tf.concat((tf.reshape(h_a, shape=[tf.shape(h_a)[0], dim]), alpha_reshape), axis=1)
            h_reshape_b = tf.concat((tf.reshape(h_b, shape=[tf.shape(h_b)[0], dim]), 1 - alpha_reshape), axis=1)
            enc_layer_list = [1200 for i in range(int_hidden_layers)] + [int_hidden_units]

            h_encode = layers.fully_connected(h_reshape_a, 'int_enc', hidden_units=enc_layer_list)
            h_encode2 = layers.fully_connected(h_reshape_b, 'int_enc', hidden_units=enc_layer_list)
            h_mix = h_encode + h_encode2

            alpha_layer_list = [1200 for i in range(int_hidden_layers)] + [alpha_dim]

            alpha_encode = layers.fully_connected(alpha_reshape, 'int_alpha', hidden_units=alpha_layer_list)
            alpha_encode2 = layers.fully_connected(1 - alpha_reshape, 'int_alpha', hidden_units=alpha_layer_list)

            alpha_encode = tf.reshape(alpha_encode, [tf.shape(alpha_encode)[0], 1, 1, 1])
            alpha_encode2 = tf.reshape(alpha_encode2, [tf.shape(alpha_encode2)[0], 1, 1, 1])

            dec_layer_list = [1200 for i in range(int_hidden_layers)] + [dim]
            h_bias = layers.fully_connected(h_mix, 'int_dec', hidden_units=dec_layer_list)

            h_bias = tf.reshape(h_bias, [tf.shape(h_bias)[0], FLAGS.latent_width, FLAGS.latent_width, latent])

            return (alpha + alpha * (1 - alpha) * alpha_encode) * h_a + \
                   ((1 - alpha) + alpha * (1 - alpha) * alpha_encode2) * h_b + \
                   alpha * (1 - alpha) * h_bias

        def disc(x):
            return tf.reduce_mean(
                layers.encoder(x, scales, depth, latent, 'disc_img'),
                axis=[1, 2, 3])

        def disc_interpolate(z):
            z = tf.reshape(z, shape=[tf.shape(z)[0], dim])
            disc_layer_list = [1200 for i in range(disc_hidden_layers)] + [dim]
            predicted_alpha = layers.fully_connected(z, 'disc_int', hidden_units=disc_layer_list)

            predicted_alpha = tf.reduce_mean(predicted_alpha, axis=[1])
            return predicted_alpha

        encode = encoder(x)
        decode = decoder(h)
        ae = decoder(encode)
        loss_rec = tf.losses.mean_squared_error(x, ae)

        encode_mix = interpolate(encode * self.n_scale, encode[::-1] * self.n_scale, alpha_h)
        encode_mix = encode_mix / self.n_scale
        my_encode_mix = interpolate(h * self.n_scale, h_b * self.n_scale, alpha_h)
        my_encode_mix = my_encode_mix / self.n_scale
        h_encode_mix = interpolate(h, h[::-1], alpha_h)
        decode_mix = decoder(encode_mix)

        loss_disc = tf.reduce_mean(tf.square(disc(ae + reg * (x - ae)))) + \
                    tf.reduce_mean(tf.square(disc(decode_mix) - alpha_h))

        alpha_noise = tf.random_uniform([tf.shape(encode)[0], 1, 1, 1], 0, 1)
        encode_mix_noise = interpolate(h * self.n_scale,
                                       encode * self.n_scale,
                                       alpha_noise)
        encode_mix_noise = encode_mix_noise / self.n_scale
        decode_mix_noise = decoder(encode_mix_noise)

        loss_disc_noise = tf.reduce_mean(tf.square(disc(decode_mix_noise) - alpha_noise))
        loss_ae_disc_noise = tf.reduce_mean(tf.square(disc(decode_mix_noise)))

        alpha_fake = 0.5
        loss_disc_fake = tf.reduce_mean(tf.square(disc(decode) - alpha_fake))
        loss_ae_disc_fake = tf.reduce_mean(tf.square(disc(decode)))

        # use h as anchor.
        loss_disc_interpolate = tf.reduce_mean(tf.square(disc_interpolate(h))) + \
                                tf.reduce_mean(tf.square(disc_interpolate(h_encode_mix) - alpha_h))
        loss_int_disc = tf.reduce_mean(tf.square(disc_interpolate(h_encode_mix)))

        loss_ae_disc = tf.reduce_mean(tf.square(disc(decode_mix)))

        encode_flat = tf.reshape(encode, [tf.shape(encode)[0], -1])
        h_flat = tf.reshape(h, [tf.shape(h)[0], -1])
        loss_mmd = tf.nn.relu(mmd2(encode_flat, h_flat))

        xops = classifiers.single_layer_classifier(tf.stop_gradient(encode), l, self.nclass)
        xloss = tf.reduce_mean(xops.loss)

        utils.HookReport.log_tensor(loss_mmd, 'loss_mmd')
        utils.HookReport.log_tensor(loss_rec, 'loss_rec')
        utils.HookReport.log_tensor(loss_ae_disc, 'loss_ae_disc')
        utils.HookReport.log_tensor(loss_disc, 'loss_disc')
        utils.HookReport.log_tensor(loss_disc_interpolate, 'loss_disc_interpolate')
        utils.HookReport.log_tensor(loss_int_disc, 'loss_int_disc')
        utils.HookReport.log_tensor(loss_disc_noise, 'loss_disc_noise')
        utils.HookReport.log_tensor(loss_ae_disc_noise, 'loss_ae_disc_noise')
        utils.HookReport.log_tensor(loss_disc_fake, 'loss_disc_fake')
        utils.HookReport.log_tensor(loss_ae_disc_fake, 'loss_ae_disc_fake')
        utils.HookReport.log_tensor(xloss, 'classify_latent')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ae_vars = tf.global_variables('ae_')
        disc_img_vars = tf.global_variables('disc_img')
        disc_int_vars = tf.global_variables('disc_int')
        int_vars = tf.global_variables('int')
        xl_vars = tf.global_variables('single_layer_classifier')

        ema = EMA(beta=beta)
        ema.apply(int_vars)

        with tf.control_dependencies(update_ops):
            train_ae = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                loss_rec + advweight * loss_ae_disc + wgt_mmd * loss_mmd +
                wgt_noise * loss_ae_disc_noise + wgt_fake * loss_ae_disc_fake,
                var_list=ae_vars)
            train_d = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                loss_disc + loss_disc_noise + loss_disc_fake, var_list=disc_img_vars)
            train_d_int = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                loss_disc_interpolate,
                var_list=disc_int_vars)
            train_int = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                loss_int_disc, tf.train.get_global_step(),
                var_list=int_vars)
            train_xl = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                xloss, tf.train.get_global_step(), var_list=xl_vars)

        ops = AEOps(x, h, h_b, l, encode, decode, ae,
                    tf.group(train_ae, train_d, train_xl),
                    train_xl, alpha_h, my_encode_mix,
                    tf.group(train_d_int, train_int), ema,
                    classify_latent=xops.output)

        n_interpolations = 16
        n_images_per_interpolation = 16

        def gen_images():
            return self.make_sample_grid_and_save(
                ops, interpolation=n_interpolations,
                height=n_images_per_interpolation)

        recon, inter, free_int, samples = tf.py_func(
            gen_images, [], [tf.float32] * 4)
        tf.summary.image('reconstruction', tf.expand_dims(recon, 0))
        tf.summary.image('interpolation', tf.expand_dims(inter, 0))
        tf.summary.image('free_int', tf.expand_dims(free_int, 0))
        tf.summary.image('samples', tf.expand_dims(samples, 0))

        return ops

    def train(self, report_kimg=1 << 6, summary_kimg=1<<6, eval_inception_kimg=1500):
        batch_size = FLAGS.batch
        with tf.Graph().as_default():
            FLAGS.total_kimg = FLAGS.train_int_kimg + FLAGS.total_kimg
            data_in = self.train_data.make_one_shot_iterator().get_next()
            global_step = tf.train.get_or_create_global_step()
            self.latent_accuracy = self.add_summary_var('latent_accuracy')
            self.mean_smoothness = self.add_summary_var('mean_smoothness')
            self.mean_distance = self.add_summary_var('mean_distance')
            some_float = tf.placeholder(tf.float32, [], 'some_float')
            update_summary_var = lambda x: tf.assign(x, some_float)
            latent_accuracy_op = update_summary_var(self.latent_accuracy)
            ops = self.model(**self.params)

            summary_hook = tf.train.SummarySaverHook(
                save_steps=(summary_kimg << 10) // batch_size,
                output_dir=self.summary_dir,
                summary_op=tf.summary.merge_all())
            stop_hook = tf.train.StopAtStepHook(last_step=1 + (FLAGS.total_kimg << 10) // batch_size)
            report_hook = utils.HookReport(report_kimg << 10, batch_size)
            run_op = lambda op, value: self.tf_sess.run(op, feed_dict={some_float: value})

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=self.checkpoint_dir,
                    hooks=[stop_hook],
                    chief_only_hooks=[report_hook, summary_hook],
                    save_checkpoint_secs=600,
                    save_summaries_steps=0) as sess:
                self.sess = sess
                self.cur_nimg = batch_size * self.tf_sess.run(global_step)
                while not sess.should_stop():
                    if self.cur_nimg < (FLAGS.train_int_kimg << 10):
                        self.train_int_step(data_in, ops)
                        self.cur_nimg = batch_size * self.tf_sess.run(global_step)
                    else:
                        self.train_step(data_in, ops)
                        self.cur_nimg = batch_size * self.tf_sess.run(global_step)

                    if self.cur_nimg % (report_kimg << 10) == 0:
                        if self.cur_nimg < (FLAGS.train_int_kimg << 10):
                            print('TRAIN INTERPOLATION ING...')
                        accuracy = self.eval_latent_accuracy(ops)
                        run_op(latent_accuracy_op, accuracy)

                    if self.cur_nimg % (eval_inception_kimg << 10) == 0:
                        inc_score = self.eval_inception_and_fid_score(ops,
                                                                      cur_image=self.cur_nimg)
                        print('In %d step, inc_score = %.4f' % (self.cur_nimg, inc_score))

    def eval_inception_and_fid_score(self, ops, num_samples=40000, batch_size=64,
                                     cur_image=0):

        dims = FLAGS.latent * FLAGS.latent_width * FLAGS.latent_width

        def batched_op(op, op_input, array):
            return np.concatenate(
                [
                    self.tf_sess.run(op, feed_dict={
                        op_input: array[x:x + batch_size]})
                    for x in range(0, array.shape[0], batch_size)
                ],
                axis=0)

        def clever_normalize(x, cmin, cmax):
            if isinstance(x, list):
                x = np.array(x, dtype=np.float)
            xmin = x.min()
            xmax = x.max()
            print('min of x is {}, max of x is {}'.format(xmin, xmax))
            x = (x - np.min(x, axis=(1, 2, 3), keepdims=True)) \
                / (np.max(x, axis=(1, 2, 3), keepdims=True) - np.min(x, axis=(1, 2, 3), keepdims=True))
            x = x * (cmax - cmin) - cmin
            return x

        def save_image(x, save_dir):
            for i in range(x.shape[0]):
                img = Image.fromarray(x[i].astype('uint8'))
                image_save_dir = save_dir + ("/%.7d.png" % (i + 1))
                img.save(image_save_dir)
            print('Save successful')

        z_sample = np.random.randn(num_samples, dims).reshape((num_samples, FLAGS.latent_width,
                                                              FLAGS.latent_width, FLAGS.latent))
        # get_real_images
        with tf.Graph().as_default():
            data_in = self.train_data.make_one_shot_iterator().get_next()
            current_size = 0
            with tf.Session() as sess_new:
                images = []
                while current_size < num_samples:
                    images.append(sess_new.run(data_in)['x'])
                    current_size += images[-1].shape[0]
                images = np.concatenate(images, axis=0)[:num_samples]

        images = clever_normalize(images, 0, 255)   # real images
        fake_image = batched_op(ops.decode, ops.h, z_sample)
        fake_image = clever_normalize(fake_image, 0, 255)

        # save the pictures
        if not os.path.exists(os.path.join(self.image_dir, str(cur_image))):
            os.mkdir(os.path.join(self.image_dir, str(cur_image)))
            print("create folder: %s" % (os.path.join(self.image_dir, str(cur_image))))

        fake_dir = os.path.join(self.image_dir, str(cur_image), 'fake')
        if not os.path.exists(fake_dir):
            os.mkdir(fake_dir)
            print("create folder: %s" % fake_dir)
        real_dir = os.path.join(self.image_dir, str(cur_image), 'real')
        if not os.path.exists(real_dir):
            os.mkdir(real_dir)
            print("create folder: %s" % real_dir)

        save_image(images, save_dir=real_dir)
        save_image(fake_image, save_dir=fake_dir)

        # get inception score
        inc_score, inc_std = get_inception_score(fake_image.transpose((0, 3, 1, 2)), splits=5)
        # fid = calculate_fid_given_paths(paths=[fake_dir, real_dir], inception_path='./Data')

        return inc_score#, #fid

    def make_sample_grid_and_save(self,
                                  ops,
                                  batch_size=10,
                                  random=4,
                                  interpolation=16,
                                  height=16,
                                  save_to_disk=True):
        # Gather images
        pool_size = random * height + 2 * height
        current_size = 0
        with tf.Graph().as_default():
            data_in = self.test_data.make_one_shot_iterator().get_next()
            with tf.Session() as sess_new:
                images = []
                while current_size < pool_size:
                    images.append(sess_new.run(data_in)['x'])
                    current_size += images[-1].shape[0]
                images = np.concatenate(images, axis=0)[:pool_size]

        def batched_op(op, op_input, array):
            return np.concatenate(
                [
                    self.tf_sess.run(op, feed_dict={
                        op_input: array[x:x + batch_size]})
                    for x in range(0, array.shape[0], batch_size)
                ],
                axis=0)

        def batched_op_list(op, op_input, array):
            return np.concatenate(
                [
                    self.tf_sess.run(op, feed_dict={
                        op_input[0]: array[0][x:x + batch_size], op_input[1]: array[1][x:x+batch_size],
                        op_input[2]: array[2][x:x + batch_size]})
                    for x in range(0, array[0].shape[0], batch_size)
                ],
                axis=0)

        # Random reconstructions
        if random:
            random_x = images[:random * height]
            random_y = batched_op(ops.ae, ops.x, random_x)
            randoms = np.concatenate([random_x, random_y], axis=2)
            image_random = utils.images_to_grid(
                randoms.reshape((height, random) + randoms.shape[1:]))
        else:
            image_random = None

        # Linear interpolations
        interpolation_x = images[-2 * height:]
        latent_x = batched_op(ops.encode, ops.x, interpolation_x)
        latents = []
        for x in range(interpolation):
            latents.append((latent_x[:height] * (interpolation - x - 1) +
                            latent_x[height:] * x) / float(interpolation - 1))
        latents = np.concatenate(latents, axis=0)
        interpolation_y = batched_op(ops.decode, ops.h, latents)
        interpolation_y = interpolation_y.reshape(
            (interpolation, height) + interpolation_y.shape[1:])
        interpolation_y = interpolation_y.transpose(1, 0, 2, 3, 4)
        image_interpolation = utils.images_to_grid(interpolation_y)

        # Free interpolation
        interpolation_x = images[-2 * height:]
        latent_x = batched_op(ops.encode, ops.x, interpolation_x)
        latents = []
        for x in range(interpolation):
            alpha = x / float(interpolation - 1) * np.ones(shape=[height, 1, 1, 1], dtype=np.float32)
            interp_latent = batched_op_list(ops.my_encode_mix, [ops.h, ops.h_b, ops.alpha_h],
                                            [latent_x[height:], latent_x[:height], alpha])
            latents.append(interp_latent)
        latents = np.concatenate(latents, axis=0)
        interpolation_y = batched_op(ops.decode, ops.h, latents)
        interpolation_y = interpolation_y.reshape(
            (interpolation, height) + interpolation_y.shape[1:])
        interpolation_y = interpolation_y.transpose(1, 0, 2, 3, 4)
        oppreserve_interpolation2 = utils.images_to_grid(interpolation_y)

        # generate synthetic images
        random_latents = np.random.standard_normal(latents.shape)

        samples_y = batched_op(ops.decode, ops.h, random_latents)
        samples_y = samples_y.reshape((interpolation, height) + samples_y.shape[1:])
        samples_y = samples_y.transpose(1, 0, 2, 3, 4)
        image_samples = utils.images_to_grid(samples_y)

        if random:
            image = np.concatenate(
                [image_random, image_interpolation, oppreserve_interpolation2,
                 image_samples], axis=1)
        else:
            image = np.concatenate(
                [image_interpolation,  oppreserve_interpolation2,
                 image_samples], axis=1)
        if save_to_disk:
            utils.save_images(utils.to_png(image), self.image_dir,
                              self.cur_nimg)

        return (image_random, image_interpolation, oppreserve_interpolation2,
                image_samples)


def main(argv):
    del argv  # Unused.
    batch = FLAGS.batch
    dataset = data.get_dataset(FLAGS.dataset, dict(batch_size=batch))
    scales = int(round(math.log(dataset.width // FLAGS.latent_width, 2)))
    model = GIDRAE2(
        dataset,
        FLAGS.train_dir,
        latent=FLAGS.latent,
        depth=FLAGS.depth,
        scales=scales,
        advweight=FLAGS.advweight,
        advdepth=FLAGS.advdepth or FLAGS.depth,
        reg=FLAGS.reg,
        int_hidden_layers=FLAGS.int_hidden_layers,
        int_hidden_units=FLAGS.int_hidden_units,
        disc_hidden_layers=FLAGS.disc_hidden_layers,
        beta=FLAGS.beta,
        wgt_mmd=FLAGS.wgt_mmd,
        use_ema=FLAGS.use_ema,
        n_scale=FLAGS.n_scale,
        wgt_noise=FLAGS.wgt_noise,
        wgt_fake=FLAGS.wgt_fake)
    model.train(summary_kimg=FLAGS.summary_kimg)


if __name__ == '__main__':
    flags.DEFINE_integer('depth', 64, 'Depth of first for convolution.')
    flags.DEFINE_integer(
        'latent', 16,
        'Latent space depth, the total latent size is the depth multiplied by '
        'latent_width ** 2.')
    flags.DEFINE_integer('latent_width', 4, 'Width of the latent space.')
    flags.DEFINE_float('advweight', 0.5, 'Adversarial weight.')
    flags.DEFINE_integer('advdepth', 0, 'Depth for adversary network.')
    flags.DEFINE_float('reg', 0.2, 'Depth for adversary network.')
    flags.DEFINE_integer('summary_kimg', 64, 'weights for mmd.')
    flags.DEFINE_integer('int_hidden_layers', 2, 'The number of layers in gi.')
    flags.DEFINE_integer('int_hidden_units', 100, 'The number of hidden units in gi.')
    flags.DEFINE_integer('disc_hidden_layers', 2, 'The number of layers in int_disc.')
    flags.DEFINE_integer('train_int_kimg', 2048, 'pretrain_steps')
    flags.DEFINE_float('beta', 0.95, 'the ema ratio of interpolation')
    flags.DEFINE_float('wgt_mmd', 1.0, 'weight for mmd loss')
    flags.DEFINE_integer('use_ema', 0, 'toggle ema.')
    flags.DEFINE_float('n_scale', 5.0, 'scale the prior when training interpolation module.')
    flags.DEFINE_float('wgt_noise', 0.1, 'weight for noise adversarial')
    flags.DEFINE_float('wgt_fake', 0.1, 'weight for fake adversarial')
    app.run(main)

'''
CUDA_VISIBLE_DEVICES=$CUDA_NUMBER python gilracai_all.py --train_dir=./TRAIN --depth=64
--latent_width=4 --latent=2 --int_hidden_layers=2 --disc_hidden_layers=2 --lr_int=1e-4
 --wgt_mmd=1.0 --MI=0 --int_method=gi --pretrain_kimg=1024 --beta=0.95 --wgt_norm=1e-4
'''