from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for every spec model: unknown fields are errors, aliases accepted by name too."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
