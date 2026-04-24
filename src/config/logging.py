from loguru import logger
import sys


def setup_logging(debug: bool = False):
    logger.remove()  # remove default handler

    if debug:
        logger.add(
            sys.stdout,
            level="DEBUG",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
        )
    else:
        logger.add(
            sys.stdout,
            level="INFO",
            format="{time} | {level} | {message}",
        )

    return logger