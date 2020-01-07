import os
import sys
import logging
import colorlog
from tqdm import tqdm
import contextlib
import inspect


class LoggingUtil:

    @staticmethod
    def setup_logger(log_dir=None, log_filename=None,
                     level_terminal=logging.INFO, level_file=logging.NOTSET,
                     reset=False):
        """
        Set the logger to log in terminal as well as to file `log_dir, log_filename` for a given level.

        :param log_dir: (string) the path to the logging directory
        :param log_filename: (string) the name of the log file
        :param level_terminal: (logging.LEVEL) the log level for the console. Default logging.INFO, everthing above info will be written to the console
        :param level_file: (logging.LEVEL) the log level for the file. Default logging.NOTSET, so everything will be written to the file
        :param reset: (boolean) if true an existing file will be cleared. Default is False, so log messages will be append
        :return: the path to the log file
        """
        # if no log_dir is given, use the current working directory
        if log_dir is None:
            log_dir = os.getcwd()

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Init logging format and level
        format_file = logging.Formatter(
            "%(asctime)s | %(filename)25s | %(lineno)4d | %(levelname)8s | %(message)s")

        format_terminal = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s | %(message)s",
            datefmt='%Y-%d-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'white',
                'INFO': 'cyan',
                'SUCCESS:': 'yellow',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white'})

        root_logger = logging.getLogger()

        # Logging to console
        # sh = logging.StreamHandler(sys.stdout)
        # since we use tqdm progress bars, we redirect logging messages
        sh = TqdmHandler()
        sh.flush()
        sh.setFormatter(format_terminal)
        root_logger.setLevel(level_terminal)
        root_logger.addHandler(sh)

        # Logging to console for errors
        handler_stderr = logging.StreamHandler(sys.stderr)
        handler_stderr.setFormatter(format_file)
        handler_stderr.setLevel(logging.CRITICAL)
        root_logger.addHandler(handler_stderr)

        # if no log_filename is given, use the filename of the script that called this function
        if log_filename is None:
            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])
            filename = module.__file__

            log_filename = os.path.splitext(filename)[0]
            log_filename = os.path.basename(log_filename)

        log_file_debug = os.path.join(log_dir, log_filename)

        logging.debug("Log file: %s" % log_file_debug)

        # if the given file exists already and reset is true
        if reset:
            # open it in write mode and clear the content of the file
            open(log_file_debug, "w").close()

        # Logging to file
        fh_debug = logging.FileHandler(log_file_debug)
        fh_debug.setFormatter(format_file)
        fh_debug.setLevel(level_file)
        root_logger.addHandler(fh_debug)

        return log_file_debug

    @staticmethod
    @contextlib.contextmanager
    def std_out_err_redirect_tqdm():
        """

        :return: (_io.TextIOWrapper) sys.stdout
        """
        orig_out_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = map(TqdmFile, orig_out_err)
            yield orig_out_err[0]
        # Relay exceptions
        except Exception as exc:
            raise exc
        # Always restore sys.stdout/err if necessary
        finally:
            sys.stdout, sys.stderr = orig_out_err


class TqdmFile(object):
    """
    File-like solution that redirects print messages to tqdm
    """
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)
