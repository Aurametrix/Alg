// pseudocode

class FeaturesExtractor:
    """Runs pretrained model without top layers on dataset and saves generated
    bottleneck features onto disk.
    """
    def __init__(self, build_fn, preprocess_fn, source,
                 target_size=(299, 299, 3), batch_size=128):        
        
        self.build_fn = build_fn
        self.preprocess_fn = preprocess_fn
        self.source = source
        self.target_size = target_size
        self.batch_size = batch_size
        
    def __call__(self, folder, filename, pool='avg'):
        model = self.build_fn(weights='imagenet', include_top=False, pooling=pool)
        stream = self.source(
            folder=folder, target_size=self.target_size,
            batch_size=self.batch_size, infinite=False)
        
        batches = []
        with tqdm.tqdm_notebook(total=stream.steps_per_epoch) as bar:
            for x_batch, y_batch in stream:
                x_preprocessed = self.preprocess_fn(x_batch)
                batch = model.predict_on_batch(x_preprocessed)
                batches.append(batch)
                bar.update(1)
                
        all_features = np.vstack(batches)
        np.save(filename, all_features)
        return filename
        
 //       ===============
 
 // SGD training

def sgd(x_train, y_train, x_valid, y_valid, variance_threshold=0.1):    
    threshold = VarianceThreshold(variance_threshold)
        
    sgd_classifier = SGDClassifier(
        alpha=1./len(x_train),
        class_weight='balanced',
        loss='log', penalty='elasticnet', 
        fit_intercept=False, tol=0.001, n_jobs=-1)
    
    bagging = BaggingClassifier(
        base_estimator=sgd_classifier,
        bootstrap_features=True,
        n_jobs=-1, max_samples=0.5, max_features=0.5)    
    
    x_thresh = threshold.fit_transform(x_train)
    bagging.fit(x_thresh, y_train)
    train_metrics = build_metrics(bagging, x_thresh, y_train)

    x_thresh = threshold.transform(x_valid)
    valid_metrics = build_metrics(bagging, x_thresh, y_valid)

    return bagging, train_metrics, valid_metrics    


def build_metrics(model, X, y):
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)
    metrics = dict(
        probs=probs,
        preds=preds,
        loss=log_loss(y, probs),
        accuracy=np.mean(preds == y))
    return namedtuple('Predictions', metrics.keys())(**metrics)    
