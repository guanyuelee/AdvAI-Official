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

"""Auto-encoder training setup.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import flags
from lib import eval, utils
import numpy as np
import tensorflow as tf
import cv2

FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', './experiments',
                    'Folder where to save training data.')
flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
flags.DEFINE_integer('batch', 64, 'Batch size.')
flags.DEFINE_string('dataset', 'lines32', 'Data to train on.')
flags.DEFINE_integer('total_kimg', 1 << 14, 'Training duration in samples.')
flags.DEFINE_integer('gen_steps', 1, 'training steps of generator per epoch.')

FLAGS = flags.FLAGS


class AEOps(object):

    def __init__(self, x, h, label, encode, decode, ae, train_op, train_op_extra,
                 classify_latent=None, train_gap=None, gap_classify=None,
                 gap_weight=None, map_encode=None):
        self.x = x
        self.h = h
        self.label = label
        self.encode = encode
        self.decode = decode
        self.ae = ae
        self.train_op = train_op
        self.train_op_extra = train_op_extra
        self.classify_latent = classify_latent
        self.train_gap=train_gap
        self.gap_classify=gap_classify
        self.gap_weight=gap_weight
        self.map_encode = map_encode


class CustomModel(object):

    def __init__(self, dataset, train_dir, **kwargs):
        self.train_data = dataset.train
        self.test_data = dataset.test
        self.train_dir = os.path.join(train_dir, dataset.name, self.experiment_name(**kwargs))
        self.height = dataset.height
        self.width = dataset.width
        self.colors = dataset.colors
        self.nclass = dataset.nclass
        self.params = kwargs
        self.sess = None
        self.cur_nimg = 0
        for dir in (self.checkpoint_dir, self.summary_dir, self.image_dir,
                    self.test_image_dir, self.train_image_dir,
                    self.train_int_image_dir, self.test_int_image_dir):
            if not os.path.exists(dir):
                os.makedirs(dir)

    def experiment_name(self, **kwargs):
        args = [x + str(y) for x, y in sorted(kwargs.items())]
        return '_'.join([self.__class__.__name__] + args)

    @property
    def test_int_image_dir(self):
        return os.path.join(self.train_dir, 'int_test')

    @property
    def train_int_image_dir(self):
        return os.path.join(self.train_dir, 'int_train')

    @property
    def test_image_dir(self):
        return os.path.join(self.train_dir, 'test')

    @property
    def train_image_dir(self):
        return os.path.join(self.train_dir, 'train')

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

    def train_step(self, data, ops):
        x = self.tf_sess.run(data)
        x, label = x['x'], x['label']
        
        shape = ops.h.get_shape().as_list()
        random_latents = np.random.standard_normal(size=(FLAGS.batch, shape[1], shape[2], shape[3]))

        self.sess.run(ops.train_op, feed_dict={ops.x: x, ops.label: label, ops.h: random_latents})


    @staticmethod
    def add_summary_var(name):
        v = tf.get_variable(name, [], trainable=False,
                            initializer=tf.initializers.zeros())
        tf.summary.scalar(name, v)
        return v


class AE(CustomModel):
    def __init__(self, dataset, train_dir, **kwargs):
        CustomModel.__init__(self, dataset, train_dir, **kwargs)
        self.latent_accuracy = 0
        self.mean_smoothness = 0
        self.mean_distance = 0
        self.eval_graph = None
        self.eval_sess = None
        self.eval_ops = None


    def eval_mode(self):
        self.eval_graph = tf.Graph()
        with self.eval_graph.as_default():
            global_step = tf.train.get_or_create_global_step()
            self.eval_ops = self.model(**self.params)
            self.eval_sess = tf.Session()
            saver = tf.train.Saver()
            print(self.checkpoint_dir)
            ckpt = utils.find_latest_checkpoint(self.checkpoint_dir)
            saver.restore(self.eval_sess, ckpt)
            tf.logging.info('Eval model %s at global_step %d',
                            self.__class__.__name__,
                            self.eval_sess.run(global_step))

    def train(self, report_kimg=1 << 6, summary_kimg=1<<6):
        batch_size = FLAGS.batch
        with tf.Graph().as_default():
            data_in = self.train_data.make_one_shot_iterator().get_next()
            global_step = tf.train.get_or_create_global_step()
            self.latent_accuracy = self.add_summary_var('latent_accuracy')
            self.gap_accuracy=self.add_summary_var('gap_accuracy')
            self.mean_smoothness = self.add_summary_var('mean_smoothness')
            self.mean_distance = self.add_summary_var('mean_distance')
            some_float = tf.placeholder(tf.float32, [], 'some_float')
            update_summary_var = lambda x: tf.assign(x, some_float)
            latent_accuracy_op = update_summary_var(self.latent_accuracy)
            gap_accuracy_op = update_summary_var(self.gap_accuracy)
            mean_smoothness_op = update_summary_var(self.mean_smoothness)
            mean_distance_op = update_summary_var(self.mean_distance)
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
                    self.train_step(data_in, ops)   
                    self.cur_nimg = batch_size * self.tf_sess.run(global_step)   

                    if FLAGS.dataset not in ('stl10_unlabeled', "celeba32"):
                        if self.cur_nimg % (report_kimg << 10) == 0:
                            if not FLAGS.dataset.startswith("celeba"):
                                accuracy = self.eval_latent_accuracy(ops)
                                #gap_accuracy=self.eval_gap_accuracy(ops)
                                run_op(latent_accuracy_op, accuracy)
                                #run_op(gap_accuracy_op, gap_accuracy)

                            if FLAGS.dataset in ('lines32', 'linesym32'):
                                mean_ds = self.eval_custom_lines32(ops)
                                run_op(mean_distance_op, mean_ds[0])
                                run_op(mean_smoothness_op, mean_ds[1])
                            elif FLAGS.dataset == 'lines32_vertical':
                                mean_ds = self.eval_custom_lines32_vertical(ops)
                                run_op(mean_distance_op, mean_ds[0])
                                run_op(mean_smoothness_op, mean_ds[1])
                            elif FLAGS.dataset == 'lines32':
                                mean_ds = self.eval_custom_lines32(ops)
                                run_op(mean_distance_op, mean_ds[0])
                                run_op(mean_smoothness_op, mean_ds[1])

    def make_sample_grid_and_save(self,
                                  ops,
                                  batch_size=10,
                                  random=12,
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
                labels = []
                while current_size < pool_size:
                    data_dict = sess_new.run(data_in)
                    images.append(data_dict["x"])
                    labels.append(data_dict["label"].argmax(1))
                    current_size += images[-1].shape[0]
                images = np.concatenate(images, axis=0)[:pool_size]
                labels = np.concatenate(labels, axis=0)[:pool_size]

        def batched_op(op, op_input, array):
            return np.concatenate(
                [
                    self.tf_sess.run(op, feed_dict={
                        op_input: array[x:x + batch_size]})
                    for x in range(0, array.shape[0], batch_size)
                ],
                axis=0)

        if random:
            random_x = images[:random * height]
            random_l = labels[:random * height]
            random_y = batched_op(ops.ae, ops.x, random_x)

            if ops.gap_weight is not None:
                encode_map = batched_op(ops.encode, ops.x, random_x)
                gap_weight = self.tf_sess.run(ops.gap_weight)  # 2x10

                encode_map, origin = utils.convert_to_heatmap(encode_map, gap_weight,
                                                              random_x.copy()[:,:,:, ::-1], random_l)

                fig_size = 32
                big_images = np.zeros(shape=[height * fig_size, random * fig_size, 3], dtype=np.uint8)
                big_encode = np.zeros(shape=[height * fig_size, random * fig_size, 3], dtype=np.uint8)
                big_encode_map = np.zeros(shape=[height * fig_size, random * 2 * fig_size, 3], dtype=np.uint8)
                for i in range(encode_map.shape[0]):
                    col = i % random
                    row = (i - col) // random
                    big_encode_map[row * fig_size: (row + 1) * fig_size,
                    col * fig_size * 2: (col + 1) * fig_size * 2] = \
                        np.concatenate((origin[i], encode_map[i]), axis=1)
                    big_images[row * fig_size: (row + 1) * fig_size,
                    col * fig_size: (col + 1) * fig_size] = origin[i]
                    big_encode[row * fig_size: (row + 1) * fig_size,
                    col * fig_size: (col + 1) * fig_size] = encode_map[i]


                encode_map = big_encode_map

                cv2.imwrite(os.path.join(self.image_dir, "heatmap%d.png" % (self.cur_nimg >> 10)),
                            encode_map)
                cv2.imwrite(os.path.join(self.image_dir, "heatmap_origin%d.png" % (self.cur_nimg >> 10)),
                            big_images)
                cv2.imwrite(os.path.join(self.image_dir, "heatmap_map%d.png" % (self.cur_nimg >> 10)),
                            big_encode)

            randoms = np.concatenate([random_x, random_y], axis=2)
            image_random = utils.images_to_grid(
                randoms.reshape((height, random) + randoms.shape[1:]))
        else:
            image_random = None

        # Interpolations
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

        # Interpolations
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

        ## generate synthetic images
        random_latents = np.random.standard_normal(latents.shape)

        samples_y = batched_op(ops.decode, ops.h, random_latents)
        samples_y = samples_y.reshape((interpolation, height) + samples_y.shape[1:])
        samples_y = samples_y.transpose(1, 0, 2, 3, 4)
        image_samples = utils.images_to_grid(samples_y)

        if random:
            image = np.concatenate(
                [image_random, image_interpolation, image_interpolation_slerp,
                 image_samples], axis=1)
        else:
            image = np.concatenate(
                [image_interpolation, image_interpolation_slerp,
                 image_samples], axis=1)
        if save_to_disk:
            utils.save_images(utils.to_png(image), self.image_dir,
                              self.cur_nimg)
        return (image_random, image_interpolation, image_interpolation_slerp,
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
                                    feed_dict={ops.x: images[p:p + batch]})
            accuracy.append((pred == labels[p:p + batch].argmax(1)))
        accuracy = 100 * np.concatenate(accuracy, axis=0).mean()
        tf.logging.info('kimg=%d  accuracy=%.2f' %
                        (self.cur_nimg >> 10, accuracy))
        return accuracy

    def eval_gap_accuracy(self, ops, batches=None):
        if ops.train_gap is None:
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
            pred = self.tf_sess.run(ops.gap_classify,
                                    feed_dict={ops.x: images[p:p + batch]})
            accuracy.append((pred == labels[p:p + batch].argmax(1)))
        accuracy = 100 * np.concatenate(accuracy, axis=0).mean()
        tf.logging.info('kimg=%d  gap accuracy=%.2f' %
                        (self.cur_nimg >> 10, accuracy))
        return accuracy


    def eval_custom_lines32(self, ops, n_interpolations=256,
                            n_images_per_interpolation=16):
        image = self.make_sample_grid_and_save(
            ops, 64, 0, n_images_per_interpolation, n_interpolations,
            save_to_disk=False)[1]
        image = image.reshape([n_interpolations, 32, n_images_per_interpolation, 32, 1])
        batch2d = image.transpose(0, 2, 1, 3, 4)
        mean_distance, mean_smoothness = eval.line_eval(batch2d)
        tf.logging.info('kimg=%d  mean_distance=%.4f  mean_smoothness=%.4f' %
                        (self.cur_nimg >> 10, mean_distance, mean_smoothness))
        return mean_distance, mean_smoothness

    def eval_custom_lines32_vertical(self, ops, n_interpolations=256,
                                     n_images_per_interpolation=16):
        image = self.make_sample_grid_and_save(
            ops, 64, 0, n_images_per_interpolation, n_interpolations,
            save_to_disk=False)[1]
        image = image.reshape([n_interpolations, 32,
                               n_images_per_interpolation, 32, 1])
        batch2d = image.transpose(0, 2, 1, 3, 4)
        mean_distance, mean_smoothness = eval.line_eval_vertical(batch2d)
        tf.logging.info('kimg=%d  mean_distance=%.4f  mean_smoothness=%.4f' %
                        (self.cur_nimg >> 10, mean_distance, mean_smoothness))
        return mean_distance, mean_smoothness

    def eval_custom_lines32(self, ops, n_interpolations=256, n_images_per_interpolation=16):
        image = self.make_sample_grid_and_save(
            ops, 64, 0, n_images_per_interpolation, n_interpolations,
            save_to_disk=False)[1]
        image = image.reshape([n_interpolations, 32, n_images_per_interpolation, 32, 1])
        batch2d = image.transpose(0, 2, 1, 3, 4)
        mean_distance, mean_smoothness = eval.line_eval(batch2d)
        tf.logging.info('kimg=%d  mean_distance=%.4f  mean_smoothness=%.4f' %
                        (self.cur_nimg >> 10, mean_distance, mean_smoothness))
        return mean_distance, mean_smoothness

    def model(self, **kwargs):
        raise NotImplementedError


class Classify(CustomModel):
    def __init__(self, dataset, train_dir, **kwargs):
        CustomModel.__init__(self, dataset, train_dir, **kwargs)
        self.test_accuracy = 0
        self.train_accuracy = 0

    def train(self, report_kimg=1 << 6):
        batch_size = FLAGS.batch
        with tf.Graph().as_default():
            data_in = self.train_data.make_one_shot_iterator().get_next()
            global_step = tf.train.get_or_create_global_step()
            self.test_accuracy = self.add_summary_var('test_accuracy')
            self.train_accuracy = self.add_summary_var('train_accuracy')
            some_float = tf.placeholder(tf.float32, [], 'some_float')
            update_summary_var = lambda x: tf.assign(x, some_float)
            test_accuracy_op = update_summary_var(self.test_accuracy)
            train_accuracy_op = update_summary_var(self.train_accuracy)
            ops = self.model(**self.params)
            summary_hook = tf.train.SummarySaverHook(
                save_steps=(report_kimg << 10) // batch_size,
                output_dir=self.summary_dir,
                summary_op=tf.summary.merge_all())
            stop_hook = tf.train.StopAtStepHook(
                last_step=1 + (FLAGS.total_kimg << 10) // batch_size)
            report_hook = utils.HookReport(report_kimg << 10, batch_size)
            run_op = lambda op, value: self.tf_sess.run(op, feed_dict={some_float: value})

            with tf.train.MonitoredTrainingSession(
                    master=FLAGS.master,
                    is_chief=(FLAGS.task == 0),
                    checkpoint_dir=self.checkpoint_dir,
                    hooks=[stop_hook],
                    chief_only_hooks=[report_hook, summary_hook],
                    save_checkpoint_secs=600,
                    save_summaries_steps=0) as sess:
                self.sess = sess
                self.cur_nimg = batch_size * self.tf_sess.run(global_step)
                while not sess.should_stop():
                    self.train_step(data_in, ops)
                    self.cur_nimg = batch_size * self.tf_sess.run(global_step)
                    if self.cur_nimg % (report_kimg << 10) == 0:
                        accuracy = self.eval_accuracy(ops, self.test_data, 'test')
                        run_op(test_accuracy_op, accuracy)
                        accuracy = self.eval_accuracy(ops, self.train_data, 'train', batches=200)
                        run_op(train_accuracy_op, accuracy)


    def process(self, x, label):
        return x, label


    def eval_accuracy(self, ops, data, name, batches=None):
        with tf.Graph().as_default():
            data_in = data.make_one_shot_iterator().get_next()
            with tf.Session() as sess_new:
                images = []
                labels = []
                while True:
                    try:
                        payload = sess_new.run(data_in)
                        x, label = self.process(payload['x'],
                                                payload['label'])
                        images.append(x)
                        assert images[-1].shape[0] == 1 or batches is not None
                        labels.append(label)
                        if len(images) == batches:
                            break
                    except tf.errors.OutOfRangeError:
                        break
            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
        accuracy = []
        batch = FLAGS.batch
        for p in range(0, images.shape[0], FLAGS.batch):
            pred = self.tf_sess.run(ops.output,
                                    feed_dict={ops.x: images[p:p + batch]})
            accuracy.append((pred == labels[p:p + batch].argmax(1)))
        accuracy = 100 * np.concatenate(accuracy, axis=0).mean()
        tf.logging.info('kimg=%d  %s accuracy=%.2f' %
                        (self.cur_nimg >> 10, name, accuracy))
        return accuracy
