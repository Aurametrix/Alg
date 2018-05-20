def linear_regression(train_X, train_Y, learn_rate=0.005):
    # define placeholders for input data
    X, Y = tf.placeholder("float"), tf.placeholder("float")

    # define the variables we're learning
    slope, intercept = tf.Variable(0.0), tf.Variable(0.0)

    # learn the slope/intercept on a ordinary least squares loss function
    loss_function = (Y - (X * slope + intercept)) ** 2

    # find the parameters slope/intercept using a basic GD optimizer
    train = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss_function)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Train the model
        for x in range(100):
            sess.run(train, feed_dict={X: train_X, Y: train_Y})

        return sess.run(slope), sess.run(intercept)
