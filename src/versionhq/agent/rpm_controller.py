import threading
import time
from typing import Optional
from typing_extensions import Self

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from versionhq._utils import Logger


class RPMController(BaseModel):
    max_rpm: Optional[int] = Field(default=None)
    _logger: Logger = PrivateAttr(default_factory=lambda: Logger(verbose=True))
    _current_rpm: int = PrivateAttr(default=0)
    _timer: Optional[threading.Timer] = PrivateAttr(default=None)
    _lock: Optional[threading.Lock] = PrivateAttr(default=None)
    _shutdown_flag: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def reset_counter(self) -> Self:
        if self.max_rpm is not None:
            if not self._shutdown_flag:
                self._lock = threading.Lock()
                self._reset_request_count()
        return self


    def _check_and_increment(self) -> bool:
        if self.max_rpm is None:
            return True

        elif self.max_rpm is not None and self._current_rpm < self.max_rpm:
            self._current_rpm += 1
            return True

        elif self.max_rpm is not None and self._current_rpm >= self.max_rpm:
            self._logger.log(level="info", message="Max RPM reached, waiting for next minute to start.", color="yellow")
            self._wait_for_next_minute()
            self._current_rpm = 1 # restart
            return True

        else:
            return False

    def check_or_wait(self) -> bool:
        if self._lock:
            with self._lock:
                return self._check_and_increment()
        else:
            return self._check_and_increment()

        return False


    def stop_rpm_counter(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None


    def _wait_for_next_minute(self) -> None:
        time.sleep(60)
        self._current_rpm = 0


    def _reset_request_count(self) -> None:
        def _reset():
            self._current_rpm = 0
            if not self._shutdown_flag:
                self._timer = threading.Timer(60.0, self._reset_request_count)
                self._timer.start()

        if self._lock:
            with self._lock:
                _reset()
        else:
            _reset()

        if self._timer:
            self._shutdown_flag = True
            self._timer.cancel()
