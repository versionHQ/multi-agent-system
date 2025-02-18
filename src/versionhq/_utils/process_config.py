from typing import Any, Dict, Type
from pydantic import BaseModel


def process_config(values_to_update: Dict[str, Any], model_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Process the config dictionary and update the values accordingly.
    Refer to the Pydantic model class for field validation.
    """

    config = values_to_update.pop("config") if "config" in values_to_update else {}

    if config:
        for k, v in config.items():
            if k not in model_class.model_fields or values_to_update.get(k) is not None:
                continue

            if isinstance(v, dict) and isinstance(values_to_update.get(k), dict):
                values_to_update[k].update(v)
            else:
                values_to_update[k] = v

    return values_to_update
