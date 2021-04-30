import tensorflow as tf
import numpy as np

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
