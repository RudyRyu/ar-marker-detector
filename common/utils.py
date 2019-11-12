import functools
import logging
import time

log = logging.getLogger(__name__)
handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter(
#                         '%(asctime)-15s [%(levelname)s] %(message)s',
#                         '%Y-%m-%d %H:%M:%S'))

handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

log.addHandler(handler)
log.setLevel(logging.INFO)

def check_time(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        s = time.time()
        value = func(*args, **kwargs)
        log.debug(f'{func.__name__} {time.time()-s:.3f} sec')
        return value
    return wrapper_timer
