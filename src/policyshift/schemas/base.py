"""Shared schema versioning for PolicyShift data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


class VersionedModel(BaseModel):
    """Pydantic base that stamps a stable schema_version on every instance."""

    schema_version: str = Field(default=SCHEMA_VERSION, description="PolicyShift data schema version")
