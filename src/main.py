import argparse
import sys
import yaml

from logzero import logger

from surrogate import SurrogateModel

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Credit score model interpretability")
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        help="path to yaml config file",
    )
    args = parser.parse_args(sys.argv[1:])

    with open(args.config, "r") as config_fd:
        config = yaml.safe_load(config_fd)

    logger.debug(f"Starts program with config: {config}")
    class_to_init = getattr(sys.modules[__name__], config["launcher"]["class"])
    try:
        class_instance = class_to_init(config)
        func = getattr(class_instance, config["launcher"]["func"])
        func()
    except:
        logger.exception("")
    logger.info("Script ends here")
