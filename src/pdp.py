from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from scipy.stats import chisquare

from logzero import logger

import joblib
import pandas as pd

class PDP:
    def __init__(self, config):
        self.config = config
        self.estimator = joblib.load(config["data"]["own_model_path"])
        self.features_to_plot = [self.config["data"]["categorical_features"]]
        logger.debug(f"{self.features_to_plot}")
        self.features_to_plot = [0, 1, 2]
        logger.debug(f"{self.features_to_plot}")
        data = pd.read_csv(config["data"]["prediction_data_path"], sep=";")
        self.X, self.y = (
            data.drop([config["data"]["y_true"], config["data"]["y_pred"]], axis=1),
            data["y_hat"],
        )
        self.pdp = PartialDependenceDisplay.from_estimator(
            self.estimator, self.X, self.features_to_plot
        )
        self.raw_values, _ = partial_dependence(self.estimator, self.X, [0])

    def plot(self):
        pass


