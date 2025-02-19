import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, PrivateAttr


class Printer:
    def print(self, content: str, color: Optional[str] = None):
        if color == "purple":
            self._print_purple(content)
        elif color == "red":
            self._print_red(content)
        elif color == "green":
            self._print_green(content)
        elif color == "blue":
            self._print_blue(content)
        elif color == "yellow":
            self._print_yellow(content)
        else:
            print(content)

    def _print_purple(self, content):
        print("\033[1m\033[95m {}\033[00m".format(content))

    def _print_green(self, content):
        print("\033[1m\033[92m {}\033[00m".format(content))

    def _print_red(self, content):
        print("\033[91m {}\033[00m".format(content))

    def _print_blue(self, content):
        print("\033[1m\033[94m {}\033[00m".format(content))

    def _print_yellow(self, content):
        print("\033[1m\033[93m {}\033[00m".format(content))


class Logger(BaseModel):
    """
    Control CLI messages.
    Color: red = error, yellow = warning, blue = info (from vhq), green = info (from third parties)
    """

    verbose: bool = Field(default=True)
    info_file_save: bool = Field(default=False, description="whether to save INFO logs")
    filename: str = Field(default=None)
    _printer: Printer = PrivateAttr(default_factory=Printer)


    def log(self, level: str, message: str, color="yellow"):
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._printer.print(f"\n{timestamp} - versionHQ [{level.upper()}]: {message}", color=color)

        self._save(level=level, message=message, filename=self.filename)


    def _save(self, level: str, message: str, filename: str = None):
        import os
        from pathlib import Path

        if level.lower() == "info" and self.info_file_save == False:
            return

        logging_level = logging.INFO
        match level:
            case "warning":
                logging_level = logging.WARNING
            case "error":
                logging_level = logging.ERROR
            case _:
                pass

        cwd = Path.cwd()
        log_file_dir = f"{cwd}/.logs"
        os.makedirs(log_file_dir, exist_ok=True)
        filename = filename if filename else self.filename if self.filename else datetime.now().strftime('%H_%M_%S_%d_%m_%Y')
        abs_dir = f"{log_file_dir}/{filename}.log"

        logging.basicConfig(filename=abs_dir, filemode='w', level=logging_level)
        logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(filename=abs_dir)
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        file_handler.setFormatter(formatter)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logger.addHandler(file_handler)

        match logging_level:
            case logging.WARNING:
                logger.warning(message)
            case logging.ERROR:
                logger.error(message)
            case _:
               logger.info(message)
