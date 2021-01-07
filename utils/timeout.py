import multiprocessing.pool
import functools
import signal
import logging

logger = logging.getLogger(__name__)


def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""

    def timeout_decorator(
        item,
    ):  # use by setting a decorator @timeout(n_sec) on top of function
        """Wrap the original function."""

        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            return async_result.get(max_timeout)

        return func_wrapper

    return timeout_decorator


def handler(signum, frame):
    raise Exception("Function call exceeded time limit.")


def timer_alarm(n_sec, handler):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(n_sec)
