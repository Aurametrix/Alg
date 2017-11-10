class Digits:

    def __init__(self):
        self.imgs = np.load('x_tr.npy') 
        self.test_imgs = np.load('x_te.npy')
        self.labels = np.load('y_tr.npy')
        self.test_labels = np.load('y_te.npy')
        self.labels = one_hot_encoded(self.labels, 10)
        self.test_labels = one_hot_encoded(self.test_labels, 10) 
        self.x_dim = [8, 8, 1]
        self.num_classes = 10

    @staticmethod
    def get_batch(batch_size, x, y): 
        """Returns a batch from the given arrays.
        """
        idx = np.random.choice(range(x.shape[0]), size=(batch_size,), replace=False)
        return x[idx], y[idx]

    def next_batch(self, batch_size, class_id=None):
        return self.get_batch(batch_size, self.imgs, self.labels)

    def test_batch(self, batch_size):
        return self.get_batch(batch_size, self.test_imgs, self.test_labels)
