import os
import logging
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")

MONITORED_CHANNEL_IDS = [
    -1001586534276,  # Философская афиша Москвы
    -1001755796778,  # Мероприятия Москва - Афиша
    -1001996407844,  # ДУХОВНАЯ МОСКВА | АФИША, МЕРОПРИЯТИЯ
    -1001575162582,  # Москва | афиша, досуг
    -1002242885172,  # Сходим сюда? / Москва
    -1003718932652,  # !!!DEBUG
]

OUTPUT_JSON_PATH = "data/events.jsonl"
