import pandas as pd
import numpy as np
<<<<<<< HEAD
=======
from scipy.stats import chisquare
>>>>>>> d0751908c0c0c81d214e6575bab8eadbb1d4eaf1

import matplotlib.pyplot as plt
import seaborn as sns
import shap

from logzero import logger
from scipy.sparse.construct import random
from scipy.stats import chisquare
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import partial_dependence
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
        self.features_idx = [i for i in range(
            len(self.categorical_features) + len(self.numerical_features)
            )
        ]  # pdp of categorical features
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
<<<<<<< HEAD
        self.plot_fairness_partial_dependence()
        # self.plot_partial_dependence()
        # self.plot_ale()
        # self.plot_shap_analysis()
=======
        self.plot_partial_dependence()
        self.plot_ale()
        self.plot_shap_analysis()
        self._statistical_parity()
>>>>>>> d0751908c0c0c81d214e6575bab8eadbb1d4eaf1

    def train_model(self) -> None:
        logger.debug(f"Initialisation of training")

        self.X_train = self.data[self.data[self.existing_pred].isna()].reset_index(
            drop=True
        )
        y_train = self.X_train[self.prediction_column]
        print(y_train)
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
        logger.debug(f"Data exported")

    def _preprocess_data(self) -> None:
        logger.debug(f"Initialisation of pre_processing")
        self.X_train_preprocessed = self.X_train.copy()
        categories = [[v for v in x.values()][0] for x in self.cat_features_order]
        logger.debug(categories)
        pipeline_preprocessing = make_column_transformer(
            ("passthrough", self.numerical_features),
            (OrdinalEncoder(categories=categories), self.categorical_features),
            ("drop", self.forbidden_columns),
        )
        self.X_train_preprocessed = pipeline_preprocessing.fit_transform(
            self.X_train_preprocessed
        )
        logger.debug(f"{pipeline_preprocessing.transformers_}")
        self.X_test = self.X_test.drop(
            columns=[self.prediction_column, self.existing_pred]
        )
        self.X_test_preprocessed = pipeline_preprocessing.transform(self.X_test)

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
        ]
        logger.debug(f"num features = {self.numerical_features}")
        logger.debug(f"num features = {self.categorical_features}")
        feature_names = self.categorical_features + self.numerical_features

        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X_test_preprocessed,
            self.features_idx,
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

    def plot_fairness_partial_dependence(self) -> None:
        # Option 1

        # logger.warn(self.features_idx)
        # self.raw_values, _ = partial_dependence(self.model, self.X_test_preprocessed, [0, 1], method='brute', grid_resolution=100)

        # logger.info("Printing PDP raw values")
        # logger.info(self.raw_values)
        # logger.info(f"Shape : {self.raw_values.shape}")
        print(self.X_test.columns)
        # mask_male = self.X_test.loc[self.X_test['Gender']==0, 'Gender']
        mask_male = self.X_test['Gender']==0
        # Option 2 
        def plot_feature(feature_name):
            pval_ls = []
            column_idx = self.X_test.columns.get_loc(feature_name)
            distinct_values = np.unique(self.X_test_preprocessed[:, column_idx])
            logger.info(f"The distinct values are {distinct_values}")
            for c_j in distinct_values:
                
                # X_modified = self.X_test.copy()
                # X_modified = self._preprocess_data(X_modified)
                X_modified = self.X_test_preprocessed.copy()
                X_modified[:, column_idx] = c_j
                print(X_modified)
                y_pred = self.model.predict(X_modified)
                print(mask_male)
                print(~mask_male)
                # y_pred_male = y_pred[mask_male]
                # y_pred_female = y_pred[~mask_male]

                # A faire avec les index
                # df_contingency = self.X_test['Gender']
                # df_contingency['Prediction'] = y_pred
                df_contingency = pd.crosstab(self.X_test['Gender'], y_pred)
                print(df_contingency)
                # count_male_loan, count_male_no_loan = df_contingency.sum(), len(mask_male) - mask_male.sum()
                # count_female_loan, count_female_no_loan = 1, 2
                # contingency_table = np.asarray([[count_male_no_loan, count_male_loan], []])

                # _, p_val = chisquare(f_obs=y_pred_male, f_exp=y_pred_female)
                # pval_ls.append(p_val)
            print(pval_ls)
            
        plot_feature("Housing")

    def plot_ice(self) -> None:
        plt.rcParams["figure.figsize"] = (20, 20)
        feature_names = self.X_test.columns
        logger.debug(f"features names = {self.X_test.columns}")

        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X_test_preprocessed,
            self.features_idx,
            feature_names=feature_names,
            kind="both",
            ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
            pd_line_kw={"color": "tab:orange", "linestyle": "--"},
        )
        display.figure_.suptitle("test")
        plt.savefig(self.config["output"]["plot_ice"])

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
            feature_names=self.X_train.drop(columns=self.forbidden_columns).columns,
            show=False,
        )
        display = plt.gcf()
        plt.savefig(self.config["output"]["plot_shap_beeswarm"])
        logger.debug(f"SHAP beeswarm done")

        plt.clf()

        display = shap.summary_plot(
            shap_values,
            self.X_train_preprocessed,
            feature_names=self.X_train.drop(columns=self.forbidden_columns).columns,
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
        pass

    def _statistical_parity(self) -> None:
        logger.debug(f"Statistical parity initialised")

        for feature in self.categorical_features + self.forbidden_columns:
            series_accepted = (
                self.X_test.loc[
                    self.X_test[self.config["output"]["y_pred_cat"]] == 1, feature
                ]
                .value_counts(normalize=True)
                .sort_index()
            )
            series_refused = (
                self.X_test.loc[
                    self.X_test[self.config["output"]["y_pred_cat"]] == 0, feature
                ]
                .value_counts(normalize=True)
                .sort_index()
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

                series_plot = self.X_test.groupby(self.config["output"]["y_pred_cat"])[
                    feature
                ].value_counts(normalize=True)
                series_plot.mul(100)
                series_plot = series_plot.rename("percent").reset_index()
                fig = sns.catplot(
                    data=pd.DataFrame(series_plot),
                    x=feature,
                    y="percent",
                    hue=self.config["output"]["y_pred_cat"],
                    kind="bar",
                )

                plt.title(
                    f"Statistical parity for {feature} test with p_value"
                    f"of {p_val:0.2f} which means that {hyptohesis_message}"
                )
                fig.savefig(
                    self.config["output"]["plot_statistical_parity"]
                    + "_"
                    + str(feature)
                )

            else:
                RaiseValueError(
                    f"Some categories are not present in both accepted and refused when splitting by {feature}"
                )

        logger.debug(f"Statistical parity finalised")


#     Step 9: Assess the fairness of your own model. Use a Pearson statistic for the following three fairness
# definitions: Statistical Parity, Conditional Statistical Parity (groups are given in the dataset), and Equal
# Odds. Discuss your results.

