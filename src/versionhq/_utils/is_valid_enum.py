from enum import Enum, IntEnum
from typing import Any


def is_valid_enum(enum: Enum | IntEnum, key: str = None, val: str | Enum | IntEnum = None) -> bool:
    if not enum: return False

    if key:
        key = key.upper()
        matched = [k for k in enum._member_map_.keys() if hasattr(enum, "_member_map_") and k == key]
        return bool(matched)

    elif val:
        match val:
            case str():
                matched = [k for k in enum._value2member_map_.keys() if hasattr(enum, "_value2member_map_") and k == val]
                return bool(matched)

            case Enum() | IntEnum():
                return val in enum

            case _:
                return False

    else: return False
