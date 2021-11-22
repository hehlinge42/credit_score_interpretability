import pandas as pd
from logzero import logger


class SurrogateModel:
    def __init__(self, config):
        self.config = config

    def test(self):
        logger.debug(f"In test function")
