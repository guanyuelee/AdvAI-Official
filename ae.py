# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Adversarial latent generalization auto-encoder.
Regularized discriminator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from absl import app
from absl import flags
from lib import data, layers, train, utils, classifiers, eval
import numpy as np

FLAGS = flags.FLAGS


class AEOps(object):

    def __init__(self, x1, x2, h, label, encode, decode, ae, train_op, train_op_extra,
                 classify_latent=None):
        self.x1 = x1
        self.x2 = x2
        self.h = h
        self.label = label
        self.encode = encode
        self.decode = decode
        self.ae = ae
        self.train_op = train_op
        self.train_op_extra = train_op_extra
        self.classify_latent = classify_latent


class AE(train.AE):
    def train_step(self, data, ops):
        x = self.tf_sess.run(data)
        x, label = x['x'], x['label']

        shape = ops.h.get_shape().as_list()
        random_latents = np.random.standard_normal(size=(FLAGS.batch, shape[1], shape[2], shape[3]))
        perm = np.random.permutation(x.shape[0])

        self.sess.run(ops.train_op, feed_dict={ops.x1: x, ops.x2: x[perm], ops.label: label, ops.h: random_latents})

        # if self.cur_nimg > ((FLAGS.total_kimg << 10) // 3):
        #     self.sess.run(ops.train_op_extra, feed_dict={ops.x: x, ops.label: label, ops.h:random_latents})

    def model(self, latent, depth, scales, advweight, advdepth, reg):
        # P x P
        x1 = tf.placeholder(tf.float32,
                           [None, self.height, self.width, self.colors], 'x')
        x2 = tf.placeholder(tf.float32,
                           [None, self.height, self.width, self.colors], 'x2')
        l = tf.placeholder(tf.float32, [None, self.nclass], 'label')
        h = tf.placeholder(
            tf.float32,
            [None, self.height >> scales, self.width >> scales, latent], 'h')

        def encoder(x):
            return layers.encoder(x, scales, depth, latent, 'ae_enc')

        def decoder(h):
            v = layers.decoder(h, scales, depth, self.colors*3, 'ae_dec')
            return v

        x = tf.concat((x1, x2, x2[::-1]), axis=3)
        encode = encoder(x)
        decode = decoder(h)
        ae = decoder(encode)
        loss_ae = tf.losses.mean_squared_error(x, ae)

        utils.HookReport.log_tensor(tf.sqrt(loss_ae) * 127.5, 'rmse')
        utils.HookReport.log_tensor(loss_ae, 'loss_ae')

        xops = classifiers.single_layer_classifier(
            tf.stop_gradient(encode), l, self.nclass)
        xloss = tf.reduce_mean(xops.loss)
        utils.HookReport.log_tensor(xloss, 'classify_latent')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ae_vars = tf.global_variables('ae_')
        xl_vars = tf.global_variables('single_layer_classifier')
        with tf.control_dependencies(update_ops):
            train_ae = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                loss_ae,
                var_list=ae_vars)
            train_xl = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                xloss, tf.train.get_global_step(), var_list=xl_vars)
        ops = AEOps(x1, x2, h, l, encode, decode, ae, tf.group(train_ae, train_xl),
                    train_xl, classify_latent=xops.output)

        n_interpolations = 16
        n_images_per_interpolation = 16

        def gen_images():
            return self.make_sample_grid_and_save(
                ops, interpolation=n_interpolations,
                height=n_images_per_interpolation)

        recon, inter, slerp, samples = tf.py_func(
            gen_images, [], [tf.float32] * 4)
        tf.summary.image('reconstruction', tf.expand_dims(recon, 0))
        tf.summary.image('interpolation', tf.expand_dims(inter, 0))
        tf.summary.image('slerp', tf.expand_dims(slerp, 0))
        tf.summary.image('samples', tf.expand_dims(samples, 0))

        if FLAGS.dataset == 'lines32':
            batched = (n_interpolations, 32, n_images_per_interpolation, 32, 1)
            batched_interp = tf.transpose(
                tf.reshape(inter, batched), [0, 2, 1, 3, 4])
            mean_distance, mean_smoothness = tf.py_func(
                eval.line_eval, [batched_interp], [tf.float32, tf.float32])
            tf.summary.scalar('mean_distance', mean_distance)
            tf.summary.scalar('mean_smoothness', mean_smoothness)

        return ops

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

        def batched_op_list(op, op_input, array):
            res = []
            for x in range(0, array[0].shape[0], batch_size):
                feed_dict = {}
                for i in range(len(op_input)):
                    feed_dict[op_input[i]] = array[i][x:x + batch_size]
                res.append(self.tf_sess.run(op, feed_dict=feed_dict))
            return np.concatenate(res, axis=0)

        def batched_op(op, op_input, array):
            return np.concatenate(
                [
                    self.tf_sess.run(op, feed_dict={
                        op_input: array[x:x + batch_size]})
                    for x in range(0, array.shape[0], batch_size)
                ],
                axis=0)

        # Random reconstructions
        if random:
            random_x = images[:random * height]
            random_y = batched_op_list(ops.ae, [ops.x1, ops.x2], [random_x, random_x])
            random_y = random_y[:, :, :, 0:3]
            randoms = np.concatenate([random_x, random_y], axis=2)
            image_random = utils.images_to_grid(
                randoms.reshape((height, random) + randoms.shape[1:]))
        else:
            image_random = None

        # Interpolations
        interpolation_x = images[-2 * height:]
        latent_x = batched_op_list(ops.encode, [ops.x1, ops.x2], [interpolation_x, interpolation_x])
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
        print(image_interpolation.shape)
        image_interpolation2 = image_interpolation[:, :, 3:6]
        image_interpolation = image_interpolation[:, :, 0:3]

        # Interpolations
        '''
        latents_slerp = []
        dots = np.sum(latent_x[:height] * latent_x[height:],
                      tuple(range(1, len(latent_x.shape))),
                      keepdims=True)
        norms = np.sum(latent_x * latent_x,
                       tuple(range(1, len(latent_x.shape))),
                       keepdims=True)
        cosine_dist = dots / np.sqrt(norms[:height] * norms[height:])
        omega = np.arccos(cosine_dist)
        for x in range(interpolation):
            t = x / float(interpolation - 1)
            latents_slerp.append(
                np.sin((1 - t) * omega) / np.sin(omega) * latent_x[:height] +
                np.sin(t * omega) / np.sin(omega) * latent_x[height:])
        latents_slerp = np.concatenate(latents_slerp, axis=0)
        interpolation_y_slerp = batched_op(ops.decode, ops.h, latents_slerp)
        interpolation_y_slerp = interpolation_y_slerp.reshape(
            (interpolation, height) + interpolation_y_slerp.shape[1:])
        interpolation_y_slerp = interpolation_y_slerp.transpose(1, 0, 2, 3, 4)
        image_interpolation_slerp = utils.images_to_grid(interpolation_y_slerp)
        '''

        ## generate synthetic images
        random_latents = np.random.standard_normal(latents.shape)

        samples_y = batched_op(ops.decode, ops.h, random_latents)
        samples_y = samples_y[:, :, :, 0:3]
        samples_y = samples_y.reshape((interpolation, height) + samples_y.shape[1:])
        samples_y = samples_y.transpose(1, 0, 2, 3, 4)
        image_samples = utils.images_to_grid(samples_y)

        if random:
            image = np.concatenate(
                [image_random, image_interpolation, image_interpolation2,
                 image_samples], axis=1)
        else:
            image = np.concatenate(
                [image_interpolation, image_interpolation2,
                 image_samples], axis=1)
        if save_to_disk:
            utils.save_images(utils.to_png(image), self.image_dir,
                              self.cur_nimg)
        return (image_random, image_interpolation, image_interpolation2,
                image_samples)

    def eval_latent_accuracy(self, ops, batches=None):
        if ops.classify_latent is None:
            return 0
        with tf.Graph().as_default():
            data_in = self.test_data.make_one_shot_iterator().get_next()
            with tf.Session() as sess_new:
                images = []
                labels = []
                while True:
                    try:
                        payload = sess_new.run(data_in)
                        images.append(payload['x'])
                        assert images[-1].shape[0] == 1 or batches is not None
                        labels.append(payload['label'])
                        if len(images) == batches:
                            break
                    except tf.errors.OutOfRangeError:
                        break
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        accuracy = []
        batch = FLAGS.batch
        for p in range(0, images.shape[0], FLAGS.batch):
            pred = self.tf_sess.run(ops.classify_latent,
                                    feed_dict={ops.x1: images[p:p + batch], ops.x2: images[p:p+batch]})
            accuracy.append((pred == labels[p:p + batch].argmax(1)))
        accuracy = 100 * np.concatenate(accuracy, axis=0).mean()
        tf.logging.info('kimg=%d  accuracy=%.2f' %
                        (self.cur_nimg >> 10, accuracy))
        return accuracy


def main(argv):
    del argv  # Unused.
    batch = FLAGS.batch
    dataset = data.get_dataset(FLAGS.dataset, dict(batch_size=batch))
    scales = int(round(math.log(dataset.width // FLAGS.latent_width, 2)))
    model = AE(
        dataset,
        FLAGS.train_dir,
        latent=FLAGS.latent,
        depth=FLAGS.depth,
        scales=scales,
        advweight=FLAGS.advweight,
        advdepth=FLAGS.advdepth or FLAGS.depth,
        reg=FLAGS.reg)
    model.train(report_kimg=FLAGS.report_kimg)


if __name__ == '__main__':
    flags.DEFINE_integer('depth', 64, 'Depth of first for convolution.')
    flags.DEFINE_integer(
        'latent', 16,
        'Latent space depth, the total latent size is the depth multiplied by '
        'latent_width ** 2.')
    flags.DEFINE_integer('latent_width', 4, 'Width of the latent space.')
    flags.DEFINE_float('advweight', 0.5, 'Adversarial weight.')
    flags.DEFINE_integer('advdepth', 0, 'Depth for adversary network.')
    flags.DEFINE_float('reg', 0.2, 'Amount of discriminator regularization.')
    flags.DEFINE_integer('report_kimg', 64, 'Depth of first for convolution.')
    app.run(main)
