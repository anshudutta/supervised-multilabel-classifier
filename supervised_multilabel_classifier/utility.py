import os
import sys
import logging


def get_file_name(relative_path):
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    return os.path.join(file_dir, relative_path)


def config_logger():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', filename='algo.log', level=logging.INFO)
    return logger
