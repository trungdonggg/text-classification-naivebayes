import numpy as np


class VotingClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):
        for clf in self.classifiers:
            clf.fit(X, y)

    def predict(self, X):
        predictions = np.asarray([clf.predict(X) for clf in self.classifiers])
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
