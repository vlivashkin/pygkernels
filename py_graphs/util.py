import logging
import sys


def configure_logging():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
