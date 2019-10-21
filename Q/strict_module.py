"""Module docstring."""
__strict__ = True
from utils import log_to_network
MY_LIST = [1, 2, 3]
MY_DICT = {x: x+1 for x in MY_LIST}
def log_calls(func):
    def _wrapped(*args, **kwargs):
        log_to_network(f"{func.__name__} called!")
        return func(*args, **kwargs)
    return _wrapped
@log_calls
def hello_world():
    log_to_network("Hello World!")
