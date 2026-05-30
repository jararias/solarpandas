
"""Annotated type definitions used by the BSRN origin interface."""

from typing import Annotated

from ...validate import ValidaRegex, ValidaRange, validate_type  # noqa: F401

type Site = Annotated[str, ValidaRegex(r'^[a-z]{3}$', parser=str.lower)]
type Year = Annotated[int, ValidaRange(ge=1980, le=2100, parser=int)]
type Month = Annotated[int, ValidaRange(ge=1, le=12, parser=int)]
type LogicalRecordName = Annotated[str, ValidaRegex(r'^LR\d{4}$', parser=str.upper)]
type DataLogicalRecordName = Annotated[str, ValidaRegex(r'^LR0[135]00$', parser=str.upper)]