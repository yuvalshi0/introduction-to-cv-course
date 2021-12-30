import logging
import sys
from time import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

shandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
shandler.setFormatter(formatter)

logger.addHandler(shandler)


def log_time(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        logger.info(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func
