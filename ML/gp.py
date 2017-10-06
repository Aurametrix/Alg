import numpy as np
import matplotlib.pyplot as pl
import tensorflow as tf
import edward as ed
from edward.models import Bernoulli, MultivariateNormalTriL
from edward.util import rbf
from edward.models import Bernoulli, Normal


X_train = np.linspace(-100, 100, 150)[:, np.newaxis]
y_train = np.array([0]*70 + [1]*80)

N = X_train.shape[0]  # number of data points
D = X_train.shape[1]  # number of features

print("NxD={}x{}".format(N, D))


X = tf.placeholder(tf.float32, [None, D])
f = MultivariateNormalTriL(loc=tf.zeros(N), scale_tril=tf.cholesky(rbf(X, lengthscale=0.5)))
y = Bernoulli(logits=f)

qf = Normal(loc=tf.Variable(tf.random_normal([N])),
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([N]))))

inference = ed.KLqp({f: qf}, data={X: X_train, y: y_train})
inference.run(n_samples=10, n_iter=500)

y_post = ed.copy(y, {f: qf})
sess = ed.get_session()
X_test = X_train #np.linspace(-2, 12, 100)[:, np.newaxis]

y_q = 0
T = 20
for i in range(T):
    y_q = y_q + sess.run(y_post.mean(), feed_dict={X: X_test})

y_q = y_q/T    

pl.scatter(X_train[:,0], y_train, c='b')
pl.plot(X_test[:,0], y_q, c='r')
pl.show()
