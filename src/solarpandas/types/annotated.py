
"""Annotated domain types and validation entry points used by solarpandas."""

import re
from dataclasses import dataclass
from typing import Annotated, get_args, get_origin


@dataclass
class ValidaRegex:
    """Validator for values constrained by a regular-expression pattern."""

    pattern: str

    def validate(self, value, annotated_type):
        """Validate a string value against the configured regex pattern."""
        if not isinstance(value, str):
            raise TypeError(f"{annotated_type.__name__} must be a string")
        if not re.match(self.pattern, value):
            raise ValueError(f"{annotated_type.__name__} must match the regex pattern: {self.pattern}")
        return value

@dataclass
class ValidaRange:
    """Validator for numeric ranges using inclusive and exclusive bounds."""

    le: float | None = None
    lt: float | None = None
    ge: float | None = None
    gt: float | None = None

    def validate(self, value, annotated_type):
        """Validate that a value can be cast to float and satisfies bounds."""
        try:
            value = float(value)
        except Exception:
            raise TypeError(f"{annotated_type.__name__} must be a number")
        if (self.le is not None) and (value > self.le):
            raise ValueError(f"{annotated_type.__name__} must be less or equal than {self.le}")
        if (self.lt is not None) and (value >= self.lt):
            raise ValueError(f"{annotated_type.__name__} must be less than {self.lt}")
        if (self.ge is not None) and (value < self.ge):
            raise ValueError(f"{annotated_type.__name__} must be greater or equal than {self.ge}")
        if (self.gt is not None) and (value <= self.gt):
            raise ValueError(f"{annotated_type.__name__} must be greater than {self.gt}")
        return value

type Latitude = Annotated[float, ValidaRange(gt=-90, lt=90)]
type Longitude = Annotated[float, ValidaRange(ge=-180, le=180)]
type Elevation = Annotated[float, ValidaRange(gt=-450, lt=8900)]

# type SiteNetwork = Annotated[str, ValidaRegex(r"^[a-z]{3}/(?:bsrn|pvps)$")]
# type TimeStep = Annotated[str, ValidaRegex(r"^\d+min$")]
# type Climate = Annotated[str, ValidaRegex(r"^[A-Ea-e]$")]
# type Network = Annotated[str, ValidaRegex(r"^(?:bsrn|pvps)$")]
# type Source = Annotated[str, ValidaRegex(r"^(?:obs|synobs|synsat)$")]
# type PVSystem = Annotated[str, ValidaRegex(r"^(?:hsat|notrack)$")]


def validate_type(value, annotated_type):
    """Validate a value against an ``Annotated`` type alias.

    Parameters
    ----------
    value : Any
        Input value to validate.
    annotated_type : Any
        Type alias defined as ``Annotated[base_type, Validator(...)]``.

    Returns
    -------
    Any
        Validated value, or ``None`` when ``value`` is ``None``.

    Examples
    --------
    >>> from solarpandas.types import Latitude, validate_type
    >>> validate_type(37.2, Latitude)
    37.2
    """
    if value is not None:
        anntype_value = annotated_type.__value__
        if not hasattr(anntype_value, "__origin__") or get_origin(anntype_value) is not Annotated:
            raise TypeError(f"{annotated_type} is not an Annotated type")
        _, validator = get_args(anntype_value)
        return validator.validate(value, annotated_type)
    return None
