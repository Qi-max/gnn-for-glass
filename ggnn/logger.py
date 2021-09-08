import os
import sys
import time
import logging


def initialize_logger(filepath='.', filename=None, level=None):
    """Initialize the default logger with stdout and file handlers.
    Args:
        filepath (str): Path to the folder where the log file will be written.
        filename (str): The log filename.
        level (int): The log level. For example logging.DEBUG.
    Returns:
        (Logger): A logging instance with customized formatter and handlers.
    """
    level = level or logging.INFO
    filename = filename + ".log"

    logger = logging.getLogger(filename)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.FileHandler(os.path.join(filepath, filename), mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(screen_handler)
    logger.addHandler(handler)
    return logger


class Logger(object):

    """Logger that can also deliver timing information.

    Parameters
    ----------
    file : str
        File object or path to the file to write to.  Or set to None for
        a logger that does nothing.
    """

    def __init__(self, file):
        if file is None:
            self.file = None
            return
        if isinstance(file, str):
            self.filename = file
            file = open(file, 'a')
        self.file = file
        self.tics = {}

    def tic(self, label=None):
        """Start a timer.

        Parameters
        ----------
        label : str
            Label for managing multiple timers.
        """
        if self.file is None:
            return
        if label:
            self.tics[label] = time.time()
        else:
            self._tic = time.time()

    def __call__(self, message, toc=None, tic=False):
        """Writes message to the log file.

        Parameters
        ---------
        message : str
            Message to be written.
        toc : bool or str
            If toc=True or toc=label, it will append timing information in
            minutes to the timer.
        tic : bool or str
            If tic=True or tic=label, will start the generic timer or a timer
            associated with label. Equivalent to self.tic(label).
        """
        if self.file is None:
            return
        dt = ''
        if toc:
            if toc is True:
                tic = self._tic
            else:
                tic = self.tics[toc]
            dt = (time.time() - tic) / 60.
            dt = ' %.1f min.' % dt
        if self.file.closed:
            self.file = open(self.filename, 'a')
        self.file.write(message + dt + '\n')
        self.file.flush()
        if tic:
            if tic is True:
                self.tic()
            else:
                self.tic(label=tic)
