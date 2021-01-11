import signal
import logging

logger = logging.getLogger(__name__)


class timeout:
    def __init__(
        self, seconds=10, error_message="Function call exceeded the time limit."
    ):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def handler(signum, frame):
    raise Exception("Function call exceeded time limit.")


def timer_alarm(n_sec, handler):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(n_sec)
