import pandas as pd
import numpy as np
from scipy.stats import chisquare

import matplotlib.pyplot as plt
import seaborn as sns
import shap

from logzero import logger
from scipy.sparse.construct import random
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

from sklearn.inspection import PartialDependenceDisplay

from PyALE import ale

shap.initjs()


class OwnClassifierModel:
    def __init__(self, config) -> None:
        self.config = config

        self.cat_features_order = self.config["data"]["ordered_ordinal_features"]
        logger.debug(
            f"self.cat_features_order = {self.cat_features_order} has type = {type(self.cat_features_order)}"
        )

        self.data = pd.read_csv(self.config["data"]["inputs"], sep=";")
        self.categorical_features = self.config["data"]["categorical_features"]
        self.numerical_features = self.config["data"]["numerical_features"]
        self.prediction_column = self.config["data"]["y_true"]
        self.existing_pred = self.config["data"]["y_pred"]
        self.test_size = self.config["data"]["test_size"]
        self.random_state = self.config["data"]["random_state"]
        self.model_path = self.config["data"]["own_model_path"]
        self.output_data_path = self.config["data"]["prediction_data_path"]
        self.customers = self.config["data"]["chosen_random_customers"]
        self.forbidden_columns = self.config["data"]["forbidden_columns"]

    def step_2_3_5_7(self) -> None:
        self.train_model()
        self.analyze_model_perfs()
        self.make_prediction()
        self.plot_partial_dependence()
        self.plot_ale()
        self.plot_shap_analysis()
        self.fairness_assessment()

    def train_model(self) -> None:
        logger.debug(f"Initialisation of training")

        self.X_train = self.data[self.data[self.existing_pred].isna()].reset_index(
            drop=True
        )
        y_train = self.X_train[self.prediction_column]
<<<<<<< HEAD
=======

>>>>>>> 2f1d9e0bab8597ada7a2279e78ff296ccc5f0a5d
        self.X_train.drop(
            columns=[self.prediction_column, self.existing_pred], inplace=True
        )
        self.X_test = self.data.dropna(axis="index").reset_index(drop=True)
        self.y_test = self.X_test[self.prediction_column]
        logger.debug(f"Separated train from test datasets")

        self._preprocess_data()
        logger.debug(f"Pre-processing done")

        self.model = XGBClassifier(
            random_state=self.random_state, use_label_encoder=False
        )

        self.model.fit(self.X_train_preprocessed, y_train)
        logger.debug(f"Model trained")

        dump(self.model, self.model_path)
        logger.debug(f"Model saved")

    def analyze_model_perfs(self) -> None:
        logger.debug(f"Initialisation of performance analyses")
        y_pred = self.model.predict(self.X_test_preprocessed)
        acc_score = accuracy_score(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred)

        full_model_perfs = (
            f"Global accuracy: {acc_score:0.2f} "
            f"with the following classification report: \n{class_report}"
        )
        logger.debug(f"Classification report obtained: {full_model_perfs}")
        perfs_txt = open(self.config["output"]["txt_perfs"], "w")
        perfs_txt.write(full_model_perfs)
        perfs_txt.close()

    def make_prediction(self) -> None:
        logger.debug(f"Initialisation of predictions")
        y_pred_scores = self.model.predict_proba(self.X_test_preprocessed)
        y_pred_cat = self.model.predict(self.X_test_preprocessed)
        logger.debug(f"Scores obtained")
        self.X_test[self.config["output"]["y_pred_proba"]] = y_pred_scores[:, 1]
        self.X_test[self.config["output"]["y_pred_cat"]] = y_pred_cat

        self.X_test.to_csv(self.output_data_path, sep=";", index=False)
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred_cat).ravel()
        logger.debug(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
        logger.debug(f"Data exported")

    def _preprocess_data(self) -> None:
        logger.debug(f"Initialisation of pre_processing")
        self.X_train_preprocessed = self.X_train.copy()

        categories = [[v for v in x.values()][0] for x in self.cat_features_order]
        pipeline_preprocessing = make_column_transformer(
            ("passthrough", self.numerical_features),
            (OrdinalEncoder(categories=categories), self.categorical_features),
        )

        self.X_train_preprocessed = self.X_train_preprocessed.drop(
            columns=self.forbidden_columns, axis=1
        )

        self.X_test_preprocessed = self.X_test.drop(
            columns=self.prediction_column, axis=1
        )
        self.X_test_preprocessed = self.X_test_preprocessed.drop(
            columns=self.existing_pred, axis=1
        )
        self.X_test_preprocessed = self.X_test_preprocessed.drop(
            columns=self.forbidden_columns, axis=1
        )

        self.X_train_preprocessed = pipeline_preprocessing.fit_transform(
            self.X_train_preprocessed
        )
        self.X_test_preprocessed = pipeline_preprocessing.transform(
            self.X_test_preprocessed
        )

    def plot_ale(self) -> None:
        fig, ax = plt.subplots(figsize=(20, 20), nrows=3, ncols=5)
        fig.subplots_adjust(hspace=0.5, wspace=0.001)
        count = len(self.numerical_features) - 1
        X_df = pd.DataFrame(
            self.X_test_preprocessed,
            columns=self.numerical_features + self.categorical_features,
        )

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
        ]

        feature_names = self.numerical_features + self.categorical_features

        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X_test_preprocessed,
            features_idx,
            feature_names=feature_names,
            kind="both",
            subsample=300,
            ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
            pd_line_kw={"color": "tab:orange", "linestyle": "--"},
        )
        display.figure_.suptitle(
            "Partial dependence of credit worthiness  of borrowers with RandomForest"
        )
        plt.savefig(self.config["output"]["plot_pdp"])

    def plot_shap_analysis(self) -> None:
        plt.rcParams["figure.figsize"] = (25, 25)
        plt.rcParams["figure.dpi"] = 50
        logger.debug(f"Initialisation of shap analysis")
        explainer = shap.Explainer(self.model)
        shap_values = explainer.shap_values(self.X_train_preprocessed)

        plt.clf()

        display = shap.summary_plot(
            shap_values,
            self.X_train_preprocessed,
            feature_names=self.X_train.drop(
                columns=self.forbidden_columns, axis=1
            ).columns,
            show=False,
        )
        display = plt.gcf()
        plt.savefig(self.config["output"]["plot_shap_beeswarm"])
        logger.debug(f"SHAP beeswarm done")

        plt.clf()

        display = shap.summary_plot(
            shap_values,
            self.X_train_preprocessed,
            feature_names=self.X_train.drop(
                columns=self.forbidden_columns, axis=1
            ).columns,
            show=False,
            plot_type="bar",
        )
        display = plt.gcf()
        plt.savefig(self.config["output"]["plot_shap_feature_importance"])
        logger.debug(f"SHAP feature importance done")

        plt.clf()

        X_train_preprocessed_for_shap = pd.DataFrame(self.X_train_preprocessed)

        X_train_preprocessed_for_shap.columns = self.X_train.drop(
            columns=self.forbidden_columns
        ).columns

        shap_values = explainer(X_train_preprocessed_for_shap)

        for i, customer_id in enumerate(self.customers):
            display = shap.plots.bar(shap_values[customer_id], show=False)
            display = plt.gcf()
            plt.savefig(self.config["output"]["plot_shap_unique_customer"][i])
            plt.clf()
        logger.debug(f"Individual shaps done")

    def fairness_assessment(self) -> None:
        self._statistical_parity()
        self._conditional_statistical_parity()

    def _statistical_parity(self) -> None:
        logger.debug(f"Statistical parity initialised")

        nb_bin = 5
        l_features = (
            self.categorical_features + self.forbidden_columns + self.numerical_features
        )

        self._plot_statistical_parity(
            l_features,
            self.numerical_features,
            self.X_test,
            nb_bin,
            "plot_statistical_parity",
        )

        logger.debug(f"Statistical parity finalised")

    @staticmethod
    def _column_values_distribution_split_by_outcome(
        df: pd.DataFrame, outcome_column: str, desired_outcome_value, feature: str
    ):
        return (
            df.loc[df[outcome_column] == desired_outcome_value, feature]
            .value_counts(normalize=True)
            .sort_index()
        )

    def _conditional_statistical_parity(self) -> None:
        logger.debug(f"Conditional statistical parity initialised")

        nb_bin = 5
        l_features_but_group = (
            self.categorical_features + self.forbidden_columns + self.numerical_features
        )
        l_features_but_group.remove("Group")

        self._plot_statistical_parity(
            l_features_but_group,
            self.numerical_features,
            self.X_test[self.X_test["Group"] == 1],
            nb_bin,
            "plot_conditional_statistical_parity_grp1",
        )
        self._plot_statistical_parity(
            l_features_but_group,
            self.numerical_features,
            self.X_test[self.X_test["Group"] == 0],
            nb_bin,
            "plot_conditional_statistical_parity_grp0",
        )

        logger.debug(f"Conditional statistical parity finished")


    def _plot_statistical_parity(
        self, l_features, numerical_features, df_test, nb_bin, output_path
    ):
        df = df_test.copy()
        for feature in l_features:
            if feature in numerical_features:
                df[feature] = pd.cut(df[feature], bins=nb_bin)
                series_accepted = self._column_values_distribution_split_by_outcome(
                    df, self.config["output"]["y_pred_cat"], 1, feature
                )
                series_refused = self._column_values_distribution_split_by_outcome(
                    df, self.config["output"]["y_pred_cat"], 0, feature
                )

            else:
                series_accepted = self._column_values_distribution_split_by_outcome(
                    df, self.config["output"]["y_pred_cat"], 1, feature
                )
                series_refused = self._column_values_distribution_split_by_outcome(
                    df, self.config["output"]["y_pred_cat"], 0, feature
                )

            if len(series_accepted) == len(series_refused):
                chi_square_test = chisquare(series_refused, series_accepted)
                p_val = chi_square_test[1]
                if p_val < 0.10:
                    hyptohesis_message = f"the model is unfair on {feature}"
                else:
                    hyptohesis_message = (
                        f"the model is not statistically unfair on {feature}"
                    )

                series_plot = df.groupby(self.config["output"]["y_pred_cat"])[
                    feature
                ].value_counts(normalize=True)
                series_plot.mul(100)
                series_plot = series_plot.rename("percent").reset_index()
                try:
                    fig = sns.catplot(
                        data=pd.DataFrame(series_plot),
                        x=feature,
                        y="percent",
                        hue=self.config["output"]["y_pred_cat"],
                        kind="bar",
                    )
                except:
                    fig = sns.catplot(
                        data=pd.DataFrame(series_plot),
                        x="level_1",
                        y="percent",
                        hue=self.config["output"]["y_pred_cat"],
                        kind="bar",
                    )
                plt.title(
                    f"Statistical parity for {feature} test with p_value"
                    f"of {p_val:0.2f} which means that {hyptohesis_message}"
                )
                fig.savefig(self.config["output"][output_path] + "_" + str(feature))

            else:
                print(
                    f"Some categories are not present in both accepted and refused when splitting by {feature}"
                )
