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
    verbose: bool = Field(default=True)
    _printer: Printer = PrivateAttr(default_factory=Printer)

    def log(self, level, message, color="yellow"):
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._printer.print(f"\n{timestamp} - versionHQ - {level.upper()}: {message}", color=color)
