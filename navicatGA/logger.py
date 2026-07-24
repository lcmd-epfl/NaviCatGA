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

    # Bare messages are unreadable once solver output, warnings and third-party
    # chatter interleave; a level/time prefix is enough to tell them apart.
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%H:%M:%S")

    if to_stdout:
        handler_2 = logging.StreamHandler(sys.stdout)
        handler_2.setFormatter(formatter)
        logger.addHandler(handler_2)

    if to_file and logger_file != "":
        handler_1 = logging.FileHandler(logger_file, "w", "utf-8")
        handler_1.setFormatter(formatter)
        logger.addHandler(handler_1)

    return logger


def close_logger(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Adds a new logging level to the logging module and the
    currently configured logging class.
    """
    if not methodName:
        methodName = levelName.lower()

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)
