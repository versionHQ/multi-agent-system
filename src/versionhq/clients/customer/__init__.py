from enum import Enum


class Status(str, Enum):
    NOT_ASSIGNED = 0
    READY_TO_DEPLOY = 1
    ACTIVE_ON_WORKFLOW = 2
    INACTIVE_ON_WORKFLOW = 3 # inactive customer
    EXIT_WITH_CONVERSION = 4
    SUSPENDED = 5
