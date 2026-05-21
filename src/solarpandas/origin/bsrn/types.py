
from typing import Annotated

from ...validate import ValidaRegex, ValidaRange, validate_type  # noqa: F401

type Site = Annotated[str, ValidaRegex(r'^[a-z]{3}$', parser=str.lower)]
type Year = Annotated[int, ValidaRange(ge=1980, le=2100, parser=int)]
type Month = Annotated[int, ValidaRange(ge=1, le=12, parser=int)]
