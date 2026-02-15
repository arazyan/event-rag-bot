from pydantic import BaseModel, Field
from datetime import datetime


class EventModel(BaseModel):
    event_id: str
    title: str
    date: datetime | None
    summary: str | None = Field(default=None, max_length=120)
    category: str


def validate_data():
    pass
