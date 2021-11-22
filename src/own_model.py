import pandas as pd
from logzero import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump


class OwnClassifierModel:
    def __init__(self, config) -> None:
        self.config = config["data"]
        self.data = pd.read_csv(self.config["inputs"], sep=";")
        self.categorical_features = self.config["categorical_features"]
        self.prediction_column = self.config["prediction_column"]
        self.test_size = self.config["test_size"]
        self.random_state = self.config["random_state"]
        self.model_path = self.config["own_model_path"]
        self.output_data_path = self.config["prediction_data_path"]

    def step_2_and_3(self) -> None:
        self.train_model()
        self.analyze_model_perfs()
        self.make_prediction()

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
        logger.debug(f"Prediction done")
        print(self.y_test)
        print(y_pred)
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
        pipeline_preprocessing = make_column_transformer(
            (
                OrdinalEncoder(),
                self.categorical_features,
            )  # TODO: check whether we want one-hot encoder instead (risk of fucking with SHAP and other methods)
        )  # TODO: check whether we need other pre-processing steps (NAs, etc.)
        self.X_train_preprocessed = pipeline_preprocessing.fit_transform(self.X_train)
        self.X_test_preprocessed = pipeline_preprocessing.transform(
            self.X_test.drop(columns=[self.prediction_column, "y_hat"])
        )
