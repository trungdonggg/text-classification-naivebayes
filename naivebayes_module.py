import numpy as np



class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = None
        self.feature_probs = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probs = {c: np.sum(y == c) / len(y) for c in self.classes}

        self.feature_probs = {c: {} for c in self.classes}
        for c in self.classes:
            class_samples = X[y == c]
            total_samples = len(class_samples)
            for feature_idx in range(X.shape[1]):
                feature_count = np.sum(class_samples[:, feature_idx])
                self.feature_probs[c][feature_idx] = feature_count / total_samples

    def predict(self, X):
        predictions = []
        for sample in X:
            doc_probs = {c: np.log(self.class_probs[c]) for c in self.classes}
            for c in self.classes:
                for feature_idx in range(X.shape[1]):
                    if sample[feature_idx] > 0:
                        doc_probs[c] += sample[feature_idx] * np.log1p(self.feature_probs[c][feature_idx])  # Avoid log(0)

            predicted_class = max(doc_probs, key=doc_probs.get)
            predictions.append(predicted_class)

        return predictions