import logging
import json
from src.config import OUTPUT_JSON_PATH
from src.utils.schema import EventModel


def append_event_to_json(event: EventModel):
    event_dict = event.model_dump()
    if event_dict.get("date"):
        event_dict["date"] = event_dict["date"].isoformat()

    with open(OUTPUT_JSON_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")

    logging.info(f"Мероприятие '{event.title}' сохранено в {OUTPUT_JSON_PATH}")
