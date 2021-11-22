import pandas as pd
from logzero import logger


class SurrogateModel:
    def __init__(self, config):
        self.config = config
        self.data = pd.read_csv(self.config["inputs"], sep=";")
        self.model = config["model"]

    def preprocess(self):
        logger.debug(f"dataset of len {len(self.data)}")
        self.data = self.data["y_hat"].dropna()
        logger.debug(f"dataset of len {len(self.data)}")

    def train(self):
        pass
