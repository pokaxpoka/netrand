import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input

from coinrun.config import Config

def impala_cnn(images, depths=[16, 32, 32]):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    use_batch_norm = Config.USE_BATCH_NORM == 1

    dropout_layer_num = [0]
    dropout_assign_ops = []

    def dropout_layer(out):
        if Config.DROPOUT > 0:
            out_shape = out.get_shape().as_list()
            num_features = np.prod(out_shape[1:])

            var_name = 'mask_' + str(dropout_layer_num[0])
            batch_seed_shape = out_shape[1:]
            batch_seed = tf.get_variable(var_name, shape=batch_seed_shape, initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
            batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
            dropout_assign_ops.append(batch_seed_assign)

            curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - Config.DROPOUT))

            curr_mask = curr_mask * (1.0 / (1.0 - Config.DROPOUT))

            out = out * curr_mask

        dropout_layer_num[0] += 1

        return out

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        out = dropout_layer(out)

        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)

        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value
        
        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = images
    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    return out, dropout_assign_ops

def nature_cnn(scaled_images, **conv_kwargs):
    """
    Model used in the paper "Human-level control through deep reinforcement learning" 
    https://www.nature.com/articles/nature14236
    """

    def activ(curr):
        return tf.nn.relu(curr)

    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

def choose_cnn(images):
    arch = Config.ARCHITECTURE
    scaled_images = tf.cast(images, tf.float32) / 255.
    dropout_assign_ops = []

    if arch == 'nature':
        out = nature_cnn(scaled_images)
    elif arch == 'impala':
        out, dropout_assign_ops = impala_cnn(scaled_images)
    elif arch == 'impalalarge':
        out, dropout_assign_ops = impala_cnn(scaled_images, depths=[32, 64, 64, 64, 64])
    else:
        assert(False)

    return out, dropout_assign_ops

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256):
        nenv = nbatch // nsteps
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            h, self.dropout_assign_ops = choose_cnn(processed_x)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, vf, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(vf, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            #
            if Config.USE_COLOR_TRANSFORM:
                out_shape = processed_x.get_shape().as_list()
                
                mask_vbox = tf.Variable(tf.zeros_like(processed_x, dtype=bool), trainable=False)
                rh = .2 # hard-coded velocity box size
                # mh = tf.cast(tf.cast(out_shape[1], dtype=tf.float32)*rh, dtype=tf.int32)
                mh = int(out_shape[1]*rh)
                mw = mh*2
                mask_vbox = mask_vbox[:,:mh,:mw].assign(tf.ones([out_shape[0], mh, mw, out_shape[-1]], dtype=bool))
                masked = tf.where(mask_vbox, x=tf.zeros_like(processed_x), y=processed_x)
                
                # tf.image.adjust_brightness vs. ImageEnhance.Brightness
                # tf version is additive while PIL version is multiplicative
                delta_brightness = tf.Variable(tf.random_uniform([], -.5, .5), trainable=False, name='randprocess_brightness')
                # tf.image.adjust_contrast vs. PIL.ImageEnhance.Contrast
                delta_contrast   = tf.Variable(tf.random_uniform([], .5, 1.5), trainable=False, name='randprocess_contrast')
                # tf.image.adjust_saturation vs. PIL.ImageEnhance.Color
                delta_saturation = tf.Variable(tf.random_uniform([], .5, 1.5), trainable=False, name='randprocess_saturation')

                processed_x1 = tf.image.adjust_brightness(masked, delta_brightness)
                processed_x1 = tf.clip_by_value(processed_x1, 0., 255.)
                processed_x1 = tf.where(mask_vbox, x=masked, y=processed_x1)
                processed_x2 = tf.image.adjust_contrast(processed_x1, delta_contrast)
                processed_x2 = tf.clip_by_value(processed_x2, 0., 255.)
                processed_x2 = tf.where(mask_vbox, x=masked, y=processed_x2)
                processed_x3 = tf.image.adjust_saturation(processed_x2, delta_saturation)
                processed_x3 = tf.clip_by_value(processed_x3, 0., 255.)
                processed_x3 = tf.where(mask_vbox, x=processed_x, y=processed_x3)
            else:
                processed_x3 = processed_x
            #
            h, self.dropout_assign_ops = choose_cnn(processed_x3)
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        
def random_impala_cnn(images, depths=[16, 32, 32]):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    use_batch_norm = Config.USE_BATCH_NORM == 1

    dropout_layer_num = [0]
    dropout_assign_ops = []

    def dropout_layer(out):
        if Config.DROPOUT > 0:
            out_shape = out.get_shape().as_list()
            num_features = np.prod(out_shape[1:])

            var_name = 'mask_' + str(dropout_layer_num[0])
            batch_seed_shape = out_shape[1:]
            batch_seed = tf.get_variable(var_name, shape=batch_seed_shape, initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
            batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
            dropout_assign_ops.append(batch_seed_assign)

            curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - Config.DROPOUT))

            curr_mask = curr_mask * (1.0 / (1.0 - Config.DROPOUT))

            out = out * curr_mask

        dropout_layer_num[0] += 1

        return out

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        out = dropout_layer(out)

        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)

        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value
        
        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = images
    
    # add random filter
    num_colors    = 3
    randcnn_depth = 3
    kernel_size   = 3
    fan_in  = num_colors    * kernel_size * kernel_size
    fan_out = randcnn_depth * kernel_size * kernel_size
    
    mask_vbox = tf.Variable(tf.zeros_like(images, dtype=bool), trainable=False)
    mask_shape = tf.shape(images)
    rh = .2 # hard-coded velocity box size
    mh = tf.cast(tf.cast(mask_shape[1], dtype=tf.float32)*rh, dtype=tf.int32)
    mw = mh*2
    mask_vbox = mask_vbox[:,:mh,:mw].assign(tf.ones([mask_shape[0], mh, mw, mask_shape[3]], dtype=bool))

    img  = tf.where(mask_vbox, x=tf.zeros_like(images), y=images)
    rand_img = tf.layers.conv2d(img, randcnn_depth, 3, padding='same', kernel_initializer=tf.initializers.glorot_normal(), trainable=False, name='randcnn')
    out = tf.where(mask_vbox, x=images, y=rand_img, name='randout')

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    return out, dropout_assign_ops

class RandomCnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        
        X, processed_x = observation_input(ob_space, nbatch)
        scaled_images = tf.cast(processed_x, tf.float32) / 255.
        mc_index = tf.placeholder(tf.int64, shape=[1], name='mc_index')
        
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):    
            h, self.dropout_assign_ops = random_impala_cnn(scaled_images)    
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)
            
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            clean_h, _ = impala_cnn(scaled_images)
            clean_vf = fc(clean_h, 'v', 1)[:,0]
            self.clean_pd, self.clean_pi = self.pdtype.pdfromlatent(clean_h, init_scale=0.01)
        
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        
        clean_a0 = self.clean_pd.sample()
        clean_neglogp0 = self.clean_pd.neglogp(clean_a0)
        
        self.initial_state = None
            
        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp
        
        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})
        
        def step_with_clean(flag, ob, *_args, **_kwargs):
            a, v, neglogp, c_a, c_v, c_neglogp \
            = sess.run([a0, vf, neglogp0, clean_a0, clean_vf, clean_neglogp0], {X:ob})
            if flag:
                return c_a, c_v, self.initial_state, c_neglogp
            else:
                return a, v, self.initial_state, neglogp
            
        def value_with_clean(flag, ob, *_args, **_kwargs):
            v, c_v = sess.run([vf, clean_vf], {X:ob})
            if flag:
                return c_v
            else:
                return v
            
        self.X = X
        self.H = h
        self.CH = clean_h
        self.vf = vf
        self.clean_vf =clean_vf
        
        self.step = step
        self.value = value
        self.step_with_clean = step_with_clean
        self.value_with_clean = value_with_clean
        
def get_policy():
    use_lstm = Config.USE_LSTM
    
    if use_lstm == 1:
        policy = LstmPolicy
    elif use_lstm == 0:
        policy = CnnPolicy
    elif use_lstm == 2:
        policy = RandomCnnPolicy
    else:
        assert(False)

    return policy
