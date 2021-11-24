import pandas as pd
import matplotlib.pyplot as plt

from logzero import logger
from scipy.sparse.construct import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump

from sklearn.inspection import PartialDependenceDisplay

from PyALE import ale


class OwnClassifierModel:

    CAT_FEATURES_ORDINAL = {
        "CreditHistory": ["A30", "A31", "A32", "A33", "A34"],
        "EmploymentDuration": [
            "A71",
            "A72",
            "A73",
            "A74",
            "A75",
        ],  # No need to challenge
        "Housing": ["A153", "A151", "A152"],
        "Purpose": [
            "A40",
            "A410",
            "A45",
            "A46",
            "A49",
            "A48",
            "A42",
            "A44",
            "A43",
            "A41",
        ],
        # "Purpose": ["A40", "A42", "A43", "A44"],
        "Savings": ["A61", "A62", "A63", "A64", "A65"],  # No need to challenge
        "Group": ["0", "1"],
        "Gender": ["0", "1"],
    }

    def __init__(self, config) -> None:
        self.config = config
        self.data = pd.read_csv(self.config["data"]["inputs"], sep=";")
        self.categorical_features = self.config["data"]["categorical_features"]
        self.numerical_features = self.config["data"]["numerical_features"]
        self.prediction_column = self.config["data"]["prediction_column"]
        self.test_size = self.config["data"]["test_size"]
        self.random_state = self.config["data"]["random_state"]
        self.model_path = self.config["data"]["own_model_path"]
        self.output_data_path = self.config["data"]["prediction_data_path"]

    def step_2_3_5_7(self) -> None:
        self.train_model()
        self.analyze_model_perfs()
        self.make_prediction()
        # self.plot_partial_dependence()
        self.plot_ale()

    def plot_ale(self) -> None:
        fig, ax = plt.subplots(figsize=(20, 20), nrows=3, ncols=5)
        fig.subplots_adjust(hspace=0.5, wspace=0.001)
        count = len(self.numerical_features) - 1
        X_df = pd.DataFrame(
            self.X_test_preprocessed,
            columns=self.numerical_features + self.categorical_features,
        )
        logger.debug(f"\n{X_df.shape}")
        for i, feature in enumerate(self.numerical_features):
            result = ale(
                X=X_df,
                model=self.model,
                feature=[feature],
                feature_type="auto",
                grid_size=50,
                include_CI=True,
                C=0.95,
            )
            plt.savefig(self.config["output"]["plot_ale"] + "_" + str(feature) + ".png")

        for i, feature in enumerate(self.categorical_features):
            result = ale(
                X=X_df,
                model=self.model,
                feature=[feature],
                feature_type="auto",
                grid_size=50,
                include_CI=True,
                C=0.95,
            )
            plt.savefig(self.config["output"]["plot_ale"] + "_" + str(feature) + ".png")

    def plot_partial_dependence(self) -> None:
        plt.rcParams["figure.figsize"] = (20, 20)

        features_idx = [
            i
            for i in range(
                len(self.categorical_features) + len(self.numerical_features)
            )
        ]  # pdp of categorical features
        feature_names = self.X_test.columns
        logger.debug(f"features names = {self.X_test.columns}")

        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X_test_preprocessed,
            features_idx,
            feature_names=feature_names,
            kind="both",
            subsample=20,
            ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
            pd_line_kw={"color": "tab:orange", "linestyle": "--"},
        )
        display.figure_.suptitle(
            "Partial dependence of credit worthiness  of borrowers with RandomForest"
        )
        plt.savefig(self.config["output"]["plot_pdp"])

    def plot_ice(self) -> None:
        plt.rcParams["figure.figsize"] = (20, 20)

        features_idx = [
            i
            for i in range(
                len(self.categorical_features) + len(self.numerical_features)
            )
        ]  # ice of categorical features
        feature_names = self.X_test.columns
        logger.debug(f"features names = {self.X_test.columns}")

        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X_test_preprocessed,
            features_idx,
            feature_names=feature_names,
            kind="both",
            ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
            pd_line_kw={"color": "tab:orange", "linestyle": "--"},
        )
        display.figure_.suptitle("test")
        plt.savefig(self.config["output"]["plot_ice"])

    def train_model(self) -> None:
        logger.debug(f"Initializing {self.__class__.__name__}")

        self.X_train = self.data[self.data["y_hat"].isna()].reset_index(drop=True)
        # TODO: put in config file for y_hat (cleaner)
        y_train = self.X_train[self.prediction_column]
        self.X_train.drop(columns=[self.prediction_column, "y_hat"], inplace=True)
        self.X_test = self.data.dropna(axis="index").reset_index(drop=True)
        self.y_test = self.X_test[self.prediction_column]
        logger.debug(f"Separated train from test datasets")

        self._preprocess_data()
        logger.debug(f"Pre-processing done")

        self.model = RandomForestClassifier(random_state=self.random_state)
        self.model.fit(self.X_train_preprocessed, y_train)
        logger.debug(f"Model trained")

        dump(self.model, self.model_path)
        logger.debug(f"Model saved")

    def analyze_model_perfs(
        self,
    ) -> None:  # TODO: decide which outputs we want and what we do with them

        logger.debug(f"Initializing {self.__class__.__name__}")
        y_pred = self.model.predict(self.X_test_preprocessed)
        acc_score = accuracy_score(
            self.y_test, y_pred
        )  # TODO: need to save it somewhere ?
        conf_matrix = confusion_matrix(
            self.y_test, y_pred
        )  # TODO: need to save it somewhere ?
        class_report = classification_report(self.y_test, y_pred)  # My favourite one
        logger.debug(f"Classification report obtained")
        print(class_report)

    def make_prediction(self) -> None:
        logger.debug(f"Initializing {self.__class__.__name__}")
        y_pred_scores = self.model.predict_proba(self.X_test_preprocessed)
        logger.debug(f"Scores obtained")
        self.X_test["y_hat_own_model"] = y_pred_scores[:, 1]

        self.X_test.to_csv(self.output_data_path, sep=";", index=False)
        logger.debug(f"Data exported")

    def _preprocess_data(self) -> None:
        logger.debug(f"Initializing {__class__.__name__}")
        self.X_train_preprocessed = self.X_train.copy()

        pipeline_preprocessing = make_column_transformer(
            ("passthrough", self.numerical_features),
            (
                OrdinalEncoder(
                    categories=[
                        x for x in OwnClassifierModel.CAT_FEATURES_ORDINAL.values()
                    ]
                ),
                self.categorical_features,
            ),
            # TODO: check whether we want one-hot encoder instead (risk of fucking with SHAP and other methods)
        )  # TODO: check whether we need other pre-processing steps (NAs, etc.)
        self.X_train_preprocessed = pipeline_preprocessing.fit_transform(
            self.X_train_preprocessed
        )
        logger.debug(f"{pipeline_preprocessing.transformers_}")
        self.X_test = self.X_test.drop(columns=[self.prediction_column, "y_hat"])
        self.X_test_preprocessed = pipeline_preprocessing.transform(self.X_test)
