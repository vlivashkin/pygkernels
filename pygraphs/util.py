import logging
import sys


def configure_logging():
    logging.basicConfig(stream=sys.stdout, format='%(levelname)s:%(message)s', level=logging.INFO)
