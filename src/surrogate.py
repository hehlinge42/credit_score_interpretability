import pandas as pd
import sys

from logzero import logger

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

from dtreeviz.trees import dtreeviz

class SurrogateModel:
    def __init__(self, config):
        self.config = config

        self.model_types_str = config["surrogate"]["models"]
        self.model_kwargs_dict = config["surrogate"]["kwargs"]
        self.model_types = []

        for model_type in self.model_types_str:
            self.model_types.append(getattr(sys.modules[__name__], model_type))

        self.model_kwargs = []
        for kwargs in self.model_kwargs_dict:
            for k, v in kwargs.items():
                self.model_kwargs.append(v)

        csv_filepath = (
            self.config["data"]["prediction_data_path"]
            if "prediction_data_path" in self.config["data"]
            else self.config["data"]["inputs"]
        )

        csv_read = (
            "Evaluating own black box model with surrogate models"
            if "prediction_data_path" in self.config["data"]
            else "Evaluating original black box model with surrogate models"
        )
        logger.info(f"{csv_read}")

        data = pd.read_csv(csv_filepath, sep=";")
        data = data.dropna()
        self.X, self.y = (
            data.drop([config["data"]["y_true"], config["data"]["y_pred"]], axis=1),
            data["y_hat"],
        )

    def preprocess(self):
        self.X = pd.get_dummies(
            self.X, columns=self.config["data"]["categorical_features"]
        )
        scaler = StandardScaler()
        self.X_fit = scaler.fit_transform(self.X)

    def train(self):
        res = {}
        self.preprocess()
        for i, model_type in enumerate(self.model_types):
            model = model_type(**self.model_kwargs[i])
            model.fit(self.X_fit, self.y)
            score = model.score(self.X_fit, self.y)
            res[self.model_types_str[i]] = score
            try:
                text_representation = tree.export_text(
                    model, feature_names=list(self.X.columns)
                )
                fig, ax = plt.subplots(figsize=(50, 50), nrows=1, ncols=1)
                tree.plot_tree(model, feature_names=list(self.X.columns), fontsize=10, filled=True)
                fig.savefig(self.config["output"]["plot_tree"])

                #tree.plot_tree(model, feature_names=list(self.X.columns), fontsize=10, filled=True).savefig(self.config["output"]["plot_tree"])
                #viz = dtreeviz(model, self.X_fit, self.y, target_name="CreditRisk (y)", feature_names=list(self.X.columns))
                #viz.save(self.config["output"]["plot_tree"])
            except:
                logger.exception("An error occured")
                pass
        logger.info(f"Model Outputs: {res}")