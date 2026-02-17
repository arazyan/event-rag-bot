from pydantic import BaseModel
from datetime import datetime


class EventModel(BaseModel):
    event_id: str
    title: str
    date: datetime | None
    summary: str | None
    category: str
