r"""Validation helpers and annotated domain types used by solarpandas.

This module provides validator classes to validate strings and numeric values,
plus convenience ``Annotated`` aliases for common geospatial and SoDA inputs.
The helper :func:`validate_type` executes validator instances attached to an
annotated alias declared with the ``type`` statement.
"""

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Annotated, Any, get_args, get_origin

# from loguru import logger


@dataclass
class ValidaRegex:
    """Validator for string patterns using regular expressions.

    This validator checks if a string matches a specified regex pattern.
    Optionally, a parser function can transform the validated string before
    returning it.

    Parameters
    ----------
    pattern : str
        Regex pattern to match. Use raw strings (for example ``r"..."``)
        for patterns containing backslashes.
    parser : Callable[[str], str] or None, default None
        Optional callable applied after successful validation.

    Examples
    --------
    >>> validator = ValidaRegex(pattern=r"^[A-Z]{3}$", parser=str.upper)
    >>> validator.validate("abc")
    'ABC'
    """
    pattern: str
    parser: Callable[[str], str] | None = None

    def validate(self, value: str) -> str:
        """Validate a string against the configured regex pattern.

        Parameters
        ----------
        value : str
            Value to validate.

        Returns
        -------
        str
            Original or parsed string when validation succeeds.

        Raises
        ------
        TypeError
            If ``value`` is not a string.
        ValueError
            If ``value`` does not match ``pattern``.
        """
        if not isinstance(value, str):
            raise TypeError(f"{value} must be a string")
        if not re.match(self.pattern, value):
            raise ValueError(f"{value} must match the regex pattern: {self.pattern}")
        if self.parser is not None:
            return self.parser(value)
        return value

@dataclass
class ValidaChoices:
    """Validator for a fixed set of string choices with fuzzy matching.

    This validator performs case-insensitive matching against a list of allowed
    values. If an exact match (ignoring case) is not found, it attempts fuzzy 
    matching to correct potential typos. A warning is issued when fuzzy matching
    is used.

    Parameters
    ----------
    choices : list[str]
        Allowed canonical values.
    parser : Callable[[str], str] or None, default None
        Optional callable applied after successful validation.

    Notes
    -----
    Matching is case-insensitive. If no direct case-insensitive match is found,
    fuzzy matching is attempted with a 0.4 similarity cutoff. Returned values
    are always canonical entries from ``choices``.

    Examples
    --------
    >>> validator = ValidaChoices(choices=["SPARTA", "BIRD"])
    >>> validator.validate("sparta")
    'SPARTA'
    """
    choices: list[str]
    parser: Callable[[str], str] | None = None

    def validate(self, value: str) -> str:
        """Validate a string against allowed canonical choices.

        Parameters
        ----------
        value : str
            Value to validate.

        Returns
        -------
        str
            Canonical value from ``choices``.

        Raises
        ------
        TypeError
            If ``value`` is not a string.
        ValueError
            If no close match is found.
        """
        if not isinstance(value, str):
            raise TypeError(f"{value} must be a string")
        case_safe_map = {choice.casefold(): choice for choice in self.choices}
        if value.casefold() not in case_safe_map:
            if not (matches := get_close_matches(value.casefold(), case_safe_map, n=1, cutoff=0.4)):
                raise ValueError(f"{value} is not among the allowed choices: {self.choices}")
            best_choice = case_safe_map[matches[0]]
            warnings.warn(f"{value} does not match the allowed choices. I took the closest one: {best_choice}")
            value = best_choice
        else:
            # Return the canonical value from choices, not the user input
            value = case_safe_map[value.casefold()]
        if self.parser is not None:
            return self.parser(value)
        return value

@dataclass
class ValidaRange:
    """Validator for numerical ranges.

    Validates that numeric values fall within specified boundaries using
    inclusive (ge/le) or exclusive (gt/lt) comparisons. Multiple constraints
    can be combined to define precise ranges.

    Parameters
    ----------
    le : float or None, default None
        Inclusive upper bound.
    lt : float or None, default None
        Exclusive upper bound.
    ge : float or None, default None
        Inclusive lower bound.
    gt : float or None, default None
        Exclusive lower bound.
    parser : Callable[[float], float] or None, default None
        Optional callable applied after successful validation.

    Notes
    -----
    String inputs are converted to ``float`` before validation.

    Examples
    --------
    >>> percentage = ValidaRange(ge=0, le=100)
    >>> percentage.validate("75.5")
    75.5
    """
    le: float | None = None  # less or equal than this
    lt: float | None = None  # less than this
    ge: float | None = None  # greater or equal than this
    gt: float | None = None  # greater than this
    parser: Callable[[float], float] | None = None

    def validate(self, value: float | int | str ) -> float:
        """Validate that a number falls within configured bounds.

        Parameters
        ----------
        value : float or int or str
            Numeric value to validate. Strings are converted to float.

        Returns
        -------
        float
            Original or parsed value.

        Raises
        ------
        TypeError
            If ``value`` cannot be converted to float.
        ValueError
            If ``value`` violates any configured constraint.
        """
        try:
            value = float(value)
        except Exception:
            raise TypeError(f"{value} must be a number")
        if (self.le is not None) and (value > self.le):
            raise ValueError(f"{value} must be less or equal than {self.le}")
        if (self.lt is not None) and (value >= self.lt):
            raise ValueError(f"{value} must be less than {self.lt}")
        if (self.ge is not None) and (value < self.ge):
            raise ValueError(f"{value} must be greater or equal than {self.ge}")
        if (self.gt is not None) and (value <= self.gt):
            raise ValueError(f"{value} must be greater than {self.gt}")
        if self.parser is not None:
            return self.parser(value)
        return value

def validate_type(value: Any, annotated_type: Any) -> Any:
    """Validate a value against an ``Annotated`` type definition.

    This function is the main entry point for type validation. It extracts the
    validator from an `Annotated` type alias and executes its `validate` method.
    
    This enables declarative type validation using Python's type hints system.

    Parameters
    ----------
    value : Any
        Value to validate.
    annotated_type : Any
        Alias defined as ``Annotated[base_type, Validator(...)]`` using the
        ``type`` statement.

    Returns
    -------
    Any
        Validated value, possibly transformed by the validator. If ``value`` is
        ``None``, ``None`` is returned.

    Raises
    ------
    TypeError
        If ``annotated_type`` is not a valid ``Annotated`` alias.
    ValueError
        If validator checks fail.

    Examples
    --------
    >>> validate_type(40.4, Latitude)
    40.4
    >>> validate_type("PT01H", SodaTimeStep)
    'PT01H'

    See Also
    --------
    Latitude
    Longitude
    Elevation
    SodaTimeStep

    Notes
    -----
    The function expects aliases created with ``type`` (PEP 695), for example
    ``type Latitude = Annotated[float, ValidaRange(...)]``.
    """
    if value is not None:
        anntype_value = annotated_type.__value__
        if not hasattr(anntype_value, "__origin__") or get_origin(anntype_value) is not Annotated:
            raise TypeError(f"{annotated_type} is not an Annotated type")
        _, validator = get_args(anntype_value)
        return validator.validate(value)
    return None


type Latitude = Annotated[float, ValidaRange(gt=-90, lt=90)]
"""Geographic latitude coordinate validator.

Validates latitude values in decimal degrees. Range: -90° < lat < 90° (exclusive).

Examples
--------
>>> from solarpandas.validate import Latitude, validate_type
>>> validate_type(40.4168, Latitude)
40.4168
"""

type Longitude = Annotated[float, ValidaRange(ge=-180, lt=180)]
"""Geographic longitude coordinate validator.

Validates longitude values in decimal degrees. Range: -180° ≤ lon < 180°.

Examples
--------
>>> from solarpandas.validate import Longitude, validate_type
>>> validate_type(-3.7038, Longitude)
-3.7038
"""

type Elevation = Annotated[float, ValidaRange(gt=-450, lt=8900)]
"""Surface elevation/altitude validator.

Validates elevation in meters above sea level. Range: -450m < elev < 8900m.
Covers from Dead Sea (-430m) to Mt. Everest (8849m).

Examples
--------
>>> from solarpandas.validate import Elevation, validate_type
>>> validate_type(667, Elevation)
667.0
"""

type SodaTimeStep = Annotated[str, ValidaChoices(["PT01M", "PT15M", "PT01H", "PT01D", "P01M"])]
"""Temporal resolution for SoDA API requests.

Notes
-----
Allowed values use ISO 8601 duration format: ``PT01M``, ``PT15M``, ``PT01H``,
``PT01D`` and ``P01M``.

Examples
--------
>>> from solarpandas.validate import SodaTimeStep, validate_type
>>> validate_type("PT01H", SodaTimeStep)
'PT01H'
"""

type SodaStream = Annotated[str, ValidaChoices(["mcclear", "cams_radiation"])]
"""Available data streams from the SoDA service.

Notes
-----
Allowed values are ``mcclear`` (McClear clear-sky model) and
``cams_radiation`` (CAMS all-sky service).

Examples
--------
>>> from solarpandas.validate import SodaStream, validate_type
>>> validate_type("mcclear", SodaStream)
'mcclear'
"""
