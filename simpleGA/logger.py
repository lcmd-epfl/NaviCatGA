import logging
import sys


def configure_logger(
    logger_file="output.log",
    logger_level="INFO",
    to_stdout=True,
    to_file=True,
    libraries_level=None,
):

    logger = logging.getLogger()
    addLoggingLevel("TRACE", logging.DEBUG - 5)
    numeric_level = getattr(logging, logger_level.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
    else:
        logger.setLevel(logging.INFO)

    if not isinstance(libraries_level, list):
        libraries_level = []

    for library, level in libraries_level:
        logging.getLogger(library).setLevel(getattr(logging, level))

    if to_stdout:
        handler_2 = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler_2)

    if to_file and logger_file != "":
        handler_1 = logging.FileHandler(logger_file, "w", "utf-8")
        logger.addHandler(handler_1)

    return logger


def close_logger(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    # if hasattr(logging, levelName):
    #    raise AttributeError("{} already defined in logging module".format(levelName))
    # if hasattr(logging, methodName):
    #    raise AttributeError("{} already defined in logging module".format(methodName))
    # if hasattr(logging.getLoggerClass(), methodName):
    #    raise AttributeError("{} already defined in logger class".format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)
