from scipy.signal import hanning
import tensorflow as tf
import numpy as np

N = 256 # FFT size
audio = np.random.rand(N, 1) * 2 - 1
w = hanning(N)

input  = tf.placeholder(tf.float32, shape=(N, 1))
window = tf.placeholder(tf.float32, shape=(N))
window_norm = tf.div(window, tf.reduce_sum(window))
windowed_input = tf.multiply(input, window_norm)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    windowed_input_val = sess.run(windowed_input, {
        window: w,
        input: audio
    })
