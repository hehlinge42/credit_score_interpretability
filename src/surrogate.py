import pandas as pd
import sys

from logzero import logger

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import f1_score

from dtreeviz.trees import dtreeviz


class SurrogateModel:
    def __init__(self, config):
        self.config = config

        self.models = config["surrogate"]["models"]
        logger.debug(self.models)
        self.model_types_str = []
        self.model_types = []
        self.model_kwargs = []
        self.categorical_features = self.config["data"]["categorical_features"]
        self.numerical_features = self.config["data"]["numerical_features"]
        for model_str, kwargs in self.models.items():
            self.model_types_str.append(model_str)
            self.model_types.append(getattr(sys.modules[__name__], model_str))
            if isinstance(kwargs, dict):
                self.model_kwargs.append(kwargs)
            else:
                self.model_kwargs.append({})

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
        logger.debug(config["data"]["y_true"])
        logger.debug(config["data"]["y_pred"])
        logger.debug(f"\n{data.columns}")
        self.X, self.y, self.y_cat = (
            data.drop([config["data"]["y_true"], config["data"]["y_pred"]], axis=1),
            data["y_hat"],
            (data["y_hat"] > config["surrogate"]["threshold"]).astype(int),
        )

    def preprocess(self):
        self.X = pd.get_dummies(
            self.X, columns=self.config["data"]["categorical_features"]
        )
        self.X_fit = self.X
        # scaler = StandardScaler()
        # self.X_fit = scaler.fit_transform(self.X)

    def evaluate_Lasso(self, model):
        logger.info(f"Getting evaluation for Lasso model")
        for i in range(len(self.X.columns)):
            logger.debug(f"Coef of feature {self.X.columns[i]}: {model.coef_[i]}")

    def evaluate_LogisticRegression(self, model):
        logger.info(f"Getting evaluation for LogisticRegression model")
        for i in range(len(self.X.columns)):
            logger.debug(f"Coef of feature {self.X.columns[i]}: {model.coef_[0][i]}")

    def evaluate_DecisionTreeRegressor(self, model):
        logger.info(f"Getting evaluation for DecisionTree model")

        plt.rcParams["figure.figsize"] = (20, 20)
        plt.rcParams["figure.dpi"] = 75
        display = tree.plot_tree(
            model, filled=True, fontsize=8.5, feature_names=list(self.X.columns),
        )
        plt.savefig("data/tree_visualisation.png")
        text_representation = tree.export_text(
            model, feature_names=list(self.X.columns)
        )

        with open("data/tree.txt", "w") as text_file:
            text_file.write(text_representation)

        logger.debug(f"\n{text_representation}")

    def evaluate_DecisionTreeClassifier(self, model):
        self.evaluate_DecisionTreeRegressor(model)

    def get_y_col(self, model_type):
        if model_type in ["DecisionTreeRegressor", "Lasso"]:
            return self.y
        return self.y_cat

    def train(self):
        res = {}
        self.preprocess()
        for i, model_type in enumerate(self.model_types):
            model = model_type(**self.model_kwargs[i])
            y = self.get_y_col(self.model_types_str[i])
            model.fit(self.X_fit, y)
            score = model.score(self.X_fit, y)
            res[self.model_types_str[i]] = score
            eval_func = getattr(self, "evaluate_" + self.model_types_str[i])
            eval_func(model)

        logger.info(f"Model Outputs: {res}")
