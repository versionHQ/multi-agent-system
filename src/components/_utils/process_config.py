from typing import Any, Dict, Type

from pydantic import BaseModel


def process_config(
    values_to_update: Dict[str, Any], model_class: Type[BaseModel]
) -> Dict[str, Any]:
    """
    Process the config dictionary and update the values accordingly.
    Refer to the Pydantic model class for field validation.
    """

    config = values_to_update.pop(
        "config", {}
    )  # to avoid duplicated operation, pop the config
    if not config:
        return values_to_update

    # copy values from config to the model's attributes if the attribute isn't already set.
    for key, value in config.items():
        if key not in model_class.model_fields or values_to_update.get(key) is not None:
            continue

        if isinstance(value, dict) and isinstance(values_to_update.get(key), dict):
            values_to_update[key].update(value)
        else:
            values_to_update[key] = value

    return values_to_update
