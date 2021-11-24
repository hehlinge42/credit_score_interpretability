from sklearn.inspection import PartialDependenceDisplay

from logzero import logger

import joblib
import pandas as pd


class ICE:
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
        self.ice = PartialDependenceDisplay.from_estimator(
            self.estimator,
            self.X, 
            self.features_to_plot,
            kind='both',
            ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
            pd_line_kw={"color": "tab:orange", "linestyle": "--"}
        )

    def plot(self):
        pass
