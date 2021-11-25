import pandas as pd
import sys

from logzero import logger

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score


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

        if "prediction_data_path" in self.config["data"]:
            csv_filepath = self.config["data"]["prediction_data_path"]
            data = pd.read_csv(csv_filepath, sep=";")
            data = data.dropna()
            self.X, self.y, self.y_cat = (
                data.drop(
                    [
                        self.config["output"]["y_pred_proba"],
                        self.config["output"]["y_pred_cat"],
                    ],
                    axis=1,
                ),
                data[self.config["output"]["y_pred_proba"]],
                data[self.config["output"]["y_pred_cat"]],
            )
        else:
            csv_filepath = self.config["data"]["inputs"]
            data = pd.read_csv(csv_filepath, sep=";")
            data = data.dropna()
            self.X, self.y, self.y_cat = (
                data.drop([config["data"]["y_pred"], config["data"]["y_true"]], axis=1),
                data[config["data"]["y_pred"]],
                (
                    data[config["data"]["y_pred"]] > config["surrogate"]["threshold"]
                ).astype(int),
            )

        csv_read = (
            "Evaluating own black box model with surrogate models"
            if "prediction_data_path" in self.config["data"]
            else "Evaluating original black box model with surrogate models"
        )
        logger.info(f"{csv_read}")

    def preprocess(self):
        self.X = pd.get_dummies(
            self.X, columns=self.config["data"]["categorical_features"]
        )
        scaler = StandardScaler()
        self.X_fit = scaler.fit_transform(self.X)

    def evaluate_Lasso(self, model, y_true):
        logger.info(f"Getting evaluation for Lasso model")
        r2 = model.score(self.X_fit, y_true)
        for i in range(len(self.X.columns)):
            logger.debug(f"Coef of feature {self.X.columns[i]}: {model.coef_[i]}")
        return r2

    def evaluate_LogisticRegression(self, model, y_true):
        logger.info(f"Getting evaluation for LogisticRegression model")
        y_pred = model.predict(self.X_fit)
        f1 = f1_score(y_pred, y_true)
        df = pd.DataFrame(data=[self.X.columns, model.coef_[0]]).transpose()
        logger.debug(f"\n{df}")
        return f1

    def evaluate_DecisionTreeRegressor(self, model, y_true):
        logger.info(f"Getting evaluation for DecisionTree model")
        r2 = model.score(self.X_fit, y_true)
        plt.rcParams["figure.figsize"] = (20, 20)
        plt.rcParams["figure.dpi"] = 75
        display = tree.plot_tree(
            model,
            filled=True,
            fontsize=8.5,
            feature_names=list(self.X.columns),
        )
        plt.savefig(self.config["output"]["tree_vis"])
        text_representation = tree.export_text(
            model, feature_names=list(self.X.columns)
        )
        with open(self.config["output"]["tree_text"], "w") as text_file:
            text_file.write(text_representation)
        logger.debug(f"\n{text_representation}")
        return r2

    def evaluate_DecisionTreeClassifier(self, model, y_true):
        y_pred = model.predict(self.X_fit)
        f1 = f1_score(y_pred, y_true)
        acc = y_true == y_pred
        logger.debug(f"acc: \n{sum(acc) / 600}")
        self.evaluate_DecisionTreeRegressor(model, y_true)
        return f1

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
            eval_func = getattr(self, "evaluate_" + self.model_types_str[i])
            score = eval_func(model, y)
            res[self.model_types_str[i]] = score

        logger.info(f"Model Outputs: {res}")
