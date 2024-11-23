import logging
import sys
from typing import Literal, Optional, Union


def get_logger(
    name: Optional[str] = None,
    level: Union[int, str] = logging.INFO,
    file_name: Optional[str] = None,
    warn: bool = True,
    mode: Literal["w", "a"] = "w",
) -> logging.Logger:
    """Gets a logger with the specified name and level. If a file name is provided, a file handler is added to the root logger.
    This logger also inherits from the root logger, so any changes to the root logger will affect this logger as well.

    Args:
        name (Optional[str]): name of the logger
        level (logging._Level, optional): logging level. Defaults to logging.INFO.
        file_name (Optional[str], optional): file name to log to as the root logger. Defaults to None.
        warn (bool, optional): warnings enables for overwriting file handlers. Defaults to True.
        mode (Literal["w", "a"], optional): whether to write or append. Defaults to "w".

    Returns:
        logging.Logger: _description_
    """

    # set the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Capture warnings using logging
    logging.captureWarnings(True)

    # add formatter
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s: %(message)s")

    # only add console handler if no one is already attached
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        # Create and add a console handler
        console_handler = logging.StreamHandler(sys.stdout)  # set stream to stdout
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file_name:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if warn:
                    logger.warning("There already exists a file handler in the logger. Removing it.")
                logger.removeHandler(handler)

        # Create and add a file handler if a file name is provided
        file_handler = logging.FileHandler(file_name, mode=mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logging.getLogger(name)
