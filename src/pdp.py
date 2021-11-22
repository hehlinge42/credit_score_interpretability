from sklearn.inspection import PartialDependenceDisplay

import joblib


class PDP:
    def __init__(self, config):
        self.config = config
        model = joblib.load(config["model"])
        self.pdp = PartialDependenceDisplay.from_estimator(X, features)
