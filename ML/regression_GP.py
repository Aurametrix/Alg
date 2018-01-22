def generate_regression_toy_data(n=50, n_test=100, x_range=15, x_range_test=20, noise_var=0.4):
    # training and test sine wave, test one has more points
    X_train = np.random.rand(n)*x_range
    X_test = np.linspace(0,x_range_test, 500)
    
    # add noise to training observations
    y_test = np.sin(X_test)
    y_train = np.sin(X_train)+np.random.randn(n)*noise_var
    
    return X_train, y_train, X_test, y_test
    
X_train, y_train, X_test, y_test = generate_regression_toy_data()
plt.figure(figsize=(16,4))
plt.plot(X_train, y_train, 'ro')
plt.plot(X_test, y_test)
plt.legend(["Noisy observations", "True model"])
plt.title("One-Dimensional Toy Regression Data")
plt.xlabel("$\mathbf{x}$")
_=plt.ylabel("$\mathbf{y}$")

# bring data into shogun representation (features are 2d-arrays, organised as column vectors)
feats_train=RealFeatures(X_train.reshape(1,len(X_train)))
feats_test=RealFeatures(X_test.reshape(1,len(X_test)))
labels_train=RegressionLabels(y_train)

# compute covariances for different kernel parameters
taus=np.asarray([.1,4.,32.])
Cs=np.zeros(((len(X_train), len(X_train), len(taus))))
for i in range(len(taus)):
    # compute unscalled kernel matrix (first parameter is maximum size in memory and not very important)
    kernel=GaussianKernel(10, taus[i])
    kernel.init(feats_train, feats_train)
    Cs[:,:,i]=kernel.get_kernel_matrix()


# plot
plt.figure(figsize=(16,5))
for i in range(len(taus)):
    plt.subplot(1,len(taus),i+1)
    plt.imshow(Cs[:,:,i], interpolation="nearest")
    plt.xlabel("Covariate index")
    plt.ylabel("Covariate index")
    _=plt.title("tau=%.1f" % taus[i])
    
plt.figure(figsize=(16,5))
plt.suptitle("Random Samples from GP prior")
for i in range(len(taus)):
    plt.subplot(1,len(taus),i+1)
    
    # sample a bunch of latent functions from the Gaussian Process
    # note these vectors are stored row-wise
    F=Statistics.sample_from_gaussian(np.zeros(len(X_train)), Cs[:,:,i], 3)
    
    for j in range(len(F)):
        # sort points to connect the dots with lines
        sorted_idx=X_train.argsort()

        plt.plot(X_train[sorted_idx], F[j,sorted_idx], '-', markersize=6)
    
    plt.xlabel("$\mathbf{x}_i$")
    plt.ylabel("$f(\mathbf{x}_i)$")
    _=plt.title("tau=%.1f" % taus[i])
    
 plt.figure(figsize=(16,5))
plt.suptitle("Random Samples from GP posterior")
for i in range(len(taus)):
    plt.subplot(1,len(taus),i+1)

    # create inference method instance with very small observation noise to make 
    inf=ExactInferenceMethod(GaussianKernel(10, taus[i]), feats_train, ZeroMean(), labels_train, GaussianLikelihood())
    
    C_post=inf.get_posterior_covariance()
    m_post=inf.get_posterior_mean()

    # sample a bunch of latent functions from the Gaussian Process
    # note these vectors are stored row-wise
    F=Statistics.sample_from_gaussian(m_post, C_post, 5)
    
    for j in range(len(F)):
        # sort points to connect the dots with lines
        sorted_idx=sorted(range(len(X_train)),key=lambda x:X_train[x])
        plt.plot(X_train[sorted_idx], F[j,sorted_idx], '-', markersize=6)
        plt.plot(X_train, y_train, 'r*')
    
    plt.xlabel("$\mathbf{x}_i$")
    plt.ylabel("$f(\mathbf{x}_i)$")
    _=plt.title("tau=%.1f" % taus[i])
    
    
   # helper function that plots predictive distribution and data
def plot_predictive_regression(X_train, y_train, X_test, y_test, means, variances):
    # evaluate predictive distribution in this range of y-values and preallocate predictive distribution
    y_values=np.linspace(-3,3)
    D=np.zeros((len(y_values), len(X_test)))
    
    # evaluate normal distribution at every prediction point (column)
    for j in range(np.shape(D)[1]):
        # create gaussian distributio instance, expects mean vector and covariance matrix, reshape
        gauss=GaussianDistribution(np.array(means[j]).reshape(1,), np.array(variances[j]).reshape(1,1))
    
        # evaluate predictive distribution for test point, method expects matrix
        D[:,j]=np.exp(gauss.log_pdf_multiple(y_values.reshape(1,len(y_values))))
    
    plt.pcolor(X_test,y_values,D)
    plt.colorbar()
    plt.contour(X_test,y_values,D)
    plt.plot(X_test,y_test, 'b', linewidth=3)
    plt.plot(X_test,means, 'm--', linewidth=3)
    plt.plot(X_train, y_train, 'ro')
    plt.legend(["Truth", "Prediction", "Data"])
    
  plt.figure(figsize=(18,10))
plt.suptitle("GP inference for different kernel widths")
for i in range(len(taus)):
    plt.subplot(len(taus),1,i+1)
    
    # create GP instance using inference method and train
    # use Shogun objects from above
    inf.set_kernel(GaussianKernel(10,taus[i]))
    gp=GaussianProcessRegression(inf)
    gp.train()
    
    # predict labels for all test data (note that this produces the same as the below mean vector)
    means = gp.apply(feats_test)
    
    # extract means and variance of predictive distribution for all test points
    means = gp.get_mean_vector(feats_test)
    variances = gp.get_variance_vector(feats_test)
    
    # note: y_predicted == means
    
    # plot predictive distribution and training data
    plot_predictive_regression(X_train, y_train, X_test, y_test, means, variances)
    _=plt.title("tau=%.1f" % taus[i])
