from enum import Enum


class GPTSizeEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GPTCUPEnvironmentEnum(str, Enum):
    BROWSER = "browser"
    MAC = "mac"
    WINDOWS = "windows"
    UNBUNTU = "ubuntu"


class GPTCUPTypeEnum(str, Enum):
    COMPUTER_CALL_OUTPUT = "computer_call_output"
    COMPUTER_USE_PREVIEW = "computer_use_preview"


class GPTFilterTypeEnum(str, Enum):
    eq = "eq"
    ne = "ne"
    gt = "gt"
    gte = "gte"
    lt = "lt"
    lte = "lte"
