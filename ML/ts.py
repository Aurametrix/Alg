# split the train and test data, maintaining the order
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]
    
# measure the root mean squared error
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))
    
    
# walk forward validation in a step by step manner
def walk_forward_validation(data, n_test):
    predictions = list()
    train, test = train_test_split(data, n_test)
    model = model_fit(train)
    history = [x for x in train] #seed history with training data
      # walk forward
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = model_predict(model, history)
        predictions.append(yhat) #store the forecast
        history.append(test[i]) #add it to history for next loop
    # estimate error
    error = measure_rmse(test, predictions)
    print(' > %.3f' % error)
    
    
# use mean or median to predict the future
def average_forecast(history, config):
    n, avg_type = config
    if avg_type is 'mean':
        return mean(history[-n:])
    return median(history[-n:])
    
# A simple way to use SARIMAX from statsmodels
def sarima_forecast(history, order, sorder, trend):
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend,
      enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]
    
# Exponential smoothing with statsmodels
def exp_smoothing_forecast(history, t,d,s,p,b,r):
    history = array(history)
    model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
    # fit model
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]
    
 
