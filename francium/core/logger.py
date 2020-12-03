import logging
import sys
from logging import Logger, Formatter, StreamHandler

LOG_LEVEL = logging.INFO


def setup_logger(name: str) -> Logger:
    """
    sets up a logger object
    Args:
        name (str): name used in the logger
    Returns:
        (Logger)
    """
    logger: Logger = logging.getLogger(f"{name}")

    if not logger.hasHandlers():
        logger.setLevel(LOG_LEVEL)  # set the logging level
        # logging format
        logger_format: Formatter = logging.Formatter(
            "[ %(asctime)s - %(name)s ] %(levelname)s: %(message)s"
        )

        stream_handler: StreamHandler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logger_format)
        stream_handler.flush = sys.stdout.flush
        logger.addHandler(stream_handler)
        logger.propagate = False
    else:
        # if there is already a logger defined then override that
        logger = logging.getLogger()
        logger.setLevel(LOG_LEVEL)  # set the logging level
        # logging format
        logger_format: Formatter = logging.Formatter(
            "francium [ %(asctime)s - %(name)s ] %(levelname)s: %(message)s"
        )
        handler = logger.handlers[0]
        handler.setFormatter(logger_format)

    return logger  # return the logger
