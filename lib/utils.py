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

"""Utilities."""

import glob
import os
import re
import time
import numpy as np
import tensorflow as tf
from lib import data
from munkres import Munkres
import cv2
from sklearn.cluster import KMeans


class HookReport(tf.train.SessionRunHook):
    """Custom reporting hook.

    Register your tensor scalars with HookReport.log_tensor(my_tensor, 'my_name').
    This hook will report their average values over report period argument
    provided to the constructed. The values are printed in the order the tensors
    were registered.

    Attributes:
      step: int, the current global step.
    """
    _REPORT_KEY = 'report'
    _ENABLE = True
    _TENSOR_NAMES = {}

    def __init__(self, period, batch_size):
        self.step = 0
        self._period = period // batch_size
        self._batch_size = batch_size
        self._sums = np.array([])
        self._count = 0
        self._step_ratio = 0
        self._start = time.time()

    @classmethod
    def disable(cls):
        class controlled_execution(object):
            def __enter__(self):
                cls._ENABLE = False
                return self

            def __exit__(self, type, value, traceback):
                cls._ENABLE = True
        return controlled_execution()

    def begin(self):
        self._count = 0
        self._start = time.time()

    def before_run(self, run_context):
        del run_context
        fetches = tf.get_collection(self._REPORT_KEY)
        fetches = fetches + [tf.train.get_global_step()]
        return tf.train.SessionRunArgs(fetches)

    def after_run(self, run_context, run_values):
        del run_context
        results = run_values.results
        # Note: sometimes the returned step is incorrect (off by one) for some
        # unknown reason.
        self.step = results[-1] + 1
        self._count += 1

        if not self._sums.size:
            self._sums = np.array(results[:-1], 'd')
        else:
            self._sums += np.array(results[:-1], 'd')

        if self.step // self._period != self._step_ratio:
            fetches = tf.get_collection(self._REPORT_KEY)
            stats = '  '.join('%s=% .8f' % (self._TENSOR_NAMES[tensor],
                                            value / self._count)
                              for tensor, value in zip(fetches, self._sums))
            stop = time.time()
            tf.logging.info('kimg=%d  %s  [%.8f img/s]' %
                            ((self.step * self._batch_size) >> 10, stats,
                             self._batch_size * self._count / (
                                         stop - self._start)))
            self._step_ratio = self.step // self._period
            self._start = stop
            self._sums *= 0
            self._count = 0

    def end(self, session=None):
        del session

    @classmethod
    def log_tensor(cls, tensor, name):
        """Adds a tensor to be reported by the hook.

        Args:
          tensor: `tensor scalar`, a value to report.
          name: string, the name to give the value in the report.

        Returns:
          None.
        """
        if cls._ENABLE:
            cls._TENSOR_NAMES[tensor] = name
            tf.add_to_collection(cls._REPORT_KEY, tensor)
            tf.summary.scalar(name, tensor)


# A dict where you can use a.b for a['b']
class ClassDict(dict):

    def __init__(self, *args, **kwargs):
        super(ClassDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def images_to_grid(images):
    """Converts a grid of images (5D tensor) to a single image.

    Args:
      images: 5D tensor (count_y, count_x, height, width, colors), grid of images.

    Returns:
      a 3D tensor image of shape (count_y * height, count_x * width, colors).
    """
    ny, nx, h, w, c = images.shape
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape([ny * h, nx * w, c])
    return images


def save_images(image, output_dir, cur_nimg, name=None):
    """Saves images to disk.

    Saves a file called 'name.png' containing the latest samples from the
     generator and a file called 'name_123.png' where 123 is the KiB of trained
     images.

    Args:
      image: 3D numpy array (height, width, colors), the image to save.
      output_dir: string, the directory where to save the image.
      cur_nimg: int, current number of images seen by training.

    Returns:
      None
    """
    if name:
        names = [name]
    else:
        names = ('name.png', 'name_%06d.png' % (cur_nimg >> 10))
    for name in names:
        with tf.gfile.Open(os.path.join(output_dir, name), 'wb') as f:
            f.write(image)


def to_png(x):
    """Convert a 3D tensor to png.

    Args:
      x: Tensor, 01C formatted input image.

    Returns:
      Tensor, 1D string representing the image in png format.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess_temp:
            x = tf.constant(x)
            y = tf.image.encode_png(
                tf.cast(
                    tf.clip_by_value(tf.round(127.5 + 127.5 * x), 0, 255),
                    tf.uint8),
                compression=9)
            return sess_temp.run(y)


def find_latest_checkpoint(dir):
    """Replacement for tf.train.latest_checkpoint.

    It does not rely on the "checkpoint" file which sometimes contains
    absolute path and is generally hard to work with when sharing files
    between users / computers.
    """
    r_step = re.compile('.*model\.ckpt-(?P<step>\d+)\.meta')
    matches = glob.glob(os.path.join(dir, 'model.ckpt-*.meta'))
    matches = [(int(r_step.match(x).group('step')), x) for x in matches]
    ckpt_file = max(matches)[1][:-5]
    return ckpt_file


def load_ae(path, target_dataset, batch, all_aes, return_dataset=False):
    r_param = re.compile('(?P<name>[a-zA-Z][a-z_]*)(?P<value>(T)|(F)|(\d+(\.\d+)?(,\d+)*))')
    folders = [x for x in os.path.abspath(path).split('/') if x]
    dataset = folders[-2]
    if dataset != target_dataset:
        tf.logging.log(tf.logging.WARN,
                       'Mismatched datasets between classfier and AE (%s, %s)',
                       target_dataset, dataset)
    print(folders)
    class_name, argpairs = folders[-1].split('_', 1)
    params = {}
    for x in r_param.findall(argpairs):
        name, value = x[:2]
        if ',' in value:
            pass
        elif value in ('T', 'F'):
            __dict__ = dict(T=True, F=False)
            value = __dict__[value]
        elif '.' in value:
            value = float(value)
        else:
            value = int(value)
        params[name] = value
    class_ = all_aes[class_name]
    dataset = data.get_dataset(dataset, dict(batch_size=batch))
    ae = class_(dataset, '/' + os.path.join(*(folders[:-2])), **params)
    if return_dataset:
        return ae, dataset
    else:
        return ae, folders[-1]


def to_one_hot(x, nclass):
    assert len(x.shape) == 1
    num = x.shape[0]
    one_hot = np.zeros([num, nclass], dtype=np.float32)
    x_index = np.linspace(0, num-1 , num).astype(np.int32)
    one_hot[x_index, x] = 1.0
    return one_hot


def coloring_image(x, parts=100):
    rand_float = np.random.rand(x.shape[0], 1, 1, x.shape[3])
    rand_float = np.floor(rand_float * parts) / parts * 4

    X_data = x + rand_float
    X_data[X_data >= 3.0] = -1 + (X_data[X_data >= 3.0] - 3.0)
    X_data[X_data > 1.0] = 1.0 - (X_data[X_data > 1.0] - 1.0)
    return X_data.astype(np.float32)


class Prior:
    saved_prior = []
    preset_prior = ['uniform', 'gaussian', 'circle', '9_circle', 'toy']

    def _get_uniform(self, batch_size=64):
        return np.random.rand(batch_size, self.dims) * 2 - 1.

    def _get_gaussian(self, batch_size=64):
        res = np.random.randn(batch_size, self.dims)

        return res

    def _get_circle(self, batch_size=64, fraction=0.6, radius=1):
        r = fraction * radius + (1 - fraction) * radius * np.random.rand(batch_size, 1)
        theta = np.pi * 2 * np.random.rand(batch_size, 1)

        x = r * np.sin(theta)
        y = r * np.cos(theta)

        return np.concatenate((x, y), axis=1)

    def _get_9_circle(self, batch_size=64, radius=1):
        x_c = np.random.randint(-1, 2, size=(batch_size, 1)).astype(np.float)
        y_c = np.random.randint(-1, 2, size=(batch_size, 1)).astype(np.float)

        r = np.random.rand(batch_size, 1) * radius
        theta = np.random.rand(batch_size, 1) * 2 * np.pi
        x = r ** 0.5 * np.sin(theta)
        y = r ** 0.5 * np.cos(theta)

        x = x + x_c * 3 * radius
        y = y + y_c * 3 * radius

        return np.concatenate((x, y), axis=1)

    def _get_toy(self, batch_size=64):
        r = np.random.rand(batch_size, 1)
        theta = np.random.rand(batch_size, 1) * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.sin(-x * y)

        return np.concatenate([x, y, z], axis=1)

    _loader = {'uniform': _get_uniform,
               'gaussian': _get_gaussian,
               'circle': _get_circle,
               '9_circle': _get_9_circle,
               'toy': _get_toy}

    def __init__(self, prior, dims=2):
        print('Prepare prior %s' % prior)
        if prior in self.preset_prior:
            self.dims = dims
            self.name = prior
            self.loader = self._loader[prior]
            self.fix = None

    def get_next(self, batch_size):
        return self.loader(self, batch_size)

    def get_points(self, npts):
        return self.loader(self, batch_size=npts)

    def get_fix_points(self, npts):
        if self.fix is None:
            self.fix = self.loader(self, batch_size=npts)

        return self.fix


def get_prior(prior, dims=2):
    return Prior(prior, dims)


def error(cluster, target_cluster, k):
    """ Compute error between cluster and target cluster
    :param cluster: proposed cluster
    :param target_cluster: target cluster
    :return: error
    """
    n = np.shape(target_cluster)[0]
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M[i][j] = np.sum(np.logical_and(cluster == i, target_cluster == j))
    m = Munkres()
    indexes = m.compute(-M)
    corresp = []
    for i in range(k):
        corresp.append(indexes[i][1])
    pred_corresp = [corresp[int(predicted)] for predicted in cluster]
    acc = np.sum(pred_corresp == target_cluster) / float(len(target_cluster))
    return acc


def convert_to_heatmap(encode, weight=None, images=None, labels=None):
    # encode = [n, 4, 4, 2]
    # weight = [2, 10]
    n, w, h, c = encode.shape
    if weight is not None:
        nclass = weight[0].shape[1]
        encode_map = np.reshape(encode, [n * w * h, c])
        encode_map = np.matmul(encode_map, weight).reshape([n, w, h, nclass])
    else:
        encode_map = encode
    encode_map = np.transpose(encode_map, [0, 3, 1, 2])   # [n, 10, 4, 4]
    encode_map = encode_map - np.min(encode_map, axis=(2, 3), keepdims=True)
    encode_map = encode_map / np.max(encode_map, axis=(2, 3), keepdims=True)
    encode_map = np.uint8(255 * encode_map)

    encode_maps = []
    for j in range(encode_map.shape[0]):
        if labels is not None:
            enc_map = cv2.resize(encode_map[j, 0, :, :], (32, 32))
        else:
            enc_map = cv2.resize(encode_map[j, labels[j], :, :], (32, 32))
        heatmap = cv2.applyColorMap(enc_map, cv2.COLORMAP_JET)
        encode_maps += [np.expand_dims(heatmap, 0)]

    encode_maps = np.concatenate(encode_maps, axis=0)
    encode_maps = encode_maps.astype(np.uint8)

    if images.shape[3] == 1:
        images = (images + 1)/ 2 * 255
        images = np.repeat(images, repeats=3, axis=3)
    else:
        images = (images + 1) / 2 * 255

    res = 0.5 * images + 0.5 * encode_maps
    images = images.astype(np.uint8)
    res = res.astype(np.uint8)
    # double = np.concatenate((images, res), axis=2)

    # print("res shape ", double.shape)
    print(images.dtype)
    print(images.max())
    print(images.min())
    print(res.dtype)
    print(res.max())
    print(res.min())

    return res, images







